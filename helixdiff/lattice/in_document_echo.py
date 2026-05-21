from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..tokenizer import ByteTokenizer


DEFAULT_ANCHOR_SIZES = [32, 24, 16, 12, 8, 6, 4]
SENTINEL_BYTE = 0


def _overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def _byte_class(byte: int | None) -> str:
    if byte is None:
        return "edge"
    value = int(byte)
    if 48 <= value <= 57:
        return "digit"
    if 65 <= value <= 90:
        return "upper"
    if 97 <= value <= 122:
        return "lower"
    if value in {9, 10, 13, 32}:
        return "space"
    if value in {39, 45, 95}:
        return "joiner"
    if 33 <= value <= 47 or 58 <= value <= 64 or 91 <= value <= 96 or 123 <= value <= 126:
        return "punct"
    return "other"


def _shape(bytes_: bytes) -> str:
    return "".join(_byte_class(byte)[0] for byte in bytes_)


def _decode_candidate(tokenizer: ByteTokenizer, span: bytes) -> str:
    return tokenizer.decode([tokenizer.byte_offset + int(byte) for byte in span])


def _ids_for_bytes(tokenizer: ByteTokenizer, span: bytes) -> list[int]:
    return [tokenizer.byte_offset + int(byte) for byte in span]


def _candidate_frequency(index_bytes: bytes, span: bytes, *, forbidden_start: int, forbidden_end: int) -> int:
    if not span:
        return 0
    count = 0
    search_at = 0
    while True:
        found = index_bytes.find(span, search_at)
        if found < 0:
            break
        if not _overlaps(found, found + len(span), forbidden_start, forbidden_end):
            count += 1
        search_at = found + 1
    return count


def redacted_document_bytes(document_text: str, span_start: int, span_end: int) -> bytes:
    """Return document bytes with the target bytes replaced before any echo indexing."""

    document_bytes = document_text.encode("utf-8", errors="replace")
    if span_start < 0 or span_end <= span_start or span_end > len(document_bytes):
        raise ValueError("invalid byte span for in-document echo")
    return (
        document_bytes[:span_start]
        + bytes([SENTINEL_BYTE]) * (span_end - span_start)
        + document_bytes[span_end:]
    )


def in_document_echo_candidates(
    *,
    tokenizer: ByteTokenizer,
    document_text: str,
    span_start: int,
    span_end: int,
    context_bytes: int = 36,
    anchor_sizes: list[int] | None = None,
    anchor_window_bytes: int = 64,
    limit: int = 128,
) -> list[dict[str, Any]]:
    """Mine repair candidates from the redacted visible bytes of the same eval document.

    The hidden target span is replaced with sentinel bytes before scanning. Every emitted
    candidate records enough provenance for the smoke test to audit target/anchor overlap.
    """

    document_bytes = document_text.encode("utf-8", errors="replace")
    index_bytes = redacted_document_bytes(document_text, span_start, span_end)
    hole_len = span_end - span_start
    if hole_len <= 0 or limit <= 0:
        return []
    sizes = anchor_sizes or DEFAULT_ANCHOR_SIZES
    target_left = document_bytes[max(0, span_start - context_bytes) : span_start]
    target_right = document_bytes[span_end : min(len(document_bytes), span_end + context_bytes)]
    if not target_left and not target_right:
        return []
    anchor_window_start = max(0, span_start - max(0, int(anchor_window_bytes)))
    anchor_window_end = min(len(document_bytes), span_end + max(0, int(anchor_window_bytes)))
    target_forbidden_start = span_start
    target_forbidden_end = span_end
    left_boundary_class = _byte_class(document_bytes[span_start - 1] if span_start > 0 else None)
    right_boundary_class = _byte_class(document_bytes[span_end] if span_end < len(document_bytes) else None)
    rows: dict[bytes, dict[str, Any]] = {}
    frequency_cache: dict[bytes, int] = {}

    def add_candidate(
        *,
        mode: str,
        candidate_start: int,
        candidate_end: int,
        source_window_start: int,
        source_window_end: int,
        anchor_size: int,
        base_score: float,
        frequency_override: int | None = None,
    ) -> None:
        if candidate_start < 0 or candidate_end > len(index_bytes) or candidate_end <= candidate_start:
            return
        if candidate_end - candidate_start != hole_len:
            return
        source_window_start = max(0, int(source_window_start))
        source_window_end = min(len(index_bytes), int(source_window_end))
        if source_window_end <= source_window_start:
            return
        overlaps_target = _overlaps(candidate_start, candidate_end, target_forbidden_start, target_forbidden_end)
        overlaps_anchor = _overlaps(source_window_start, source_window_end, anchor_window_start, anchor_window_end)
        candidate = index_bytes[candidate_start:candidate_end]
        window = index_bytes[source_window_start:source_window_end]
        if overlaps_target or overlaps_anchor or SENTINEL_BYTE in candidate or SENTINEL_BYTE in window:
            return
        left_neighbor = index_bytes[candidate_start - 1] if candidate_start > 0 else None
        right_neighbor = index_bytes[candidate_end] if candidate_end < len(index_bytes) else None
        boundary_matches = int(_byte_class(left_neighbor) == left_boundary_class) + int(
            _byte_class(right_neighbor) == right_boundary_class
        )
        distance = abs(candidate_start - span_start)
        frequency = frequency_override
        if frequency is None:
            frequency = frequency_cache.get(candidate)
        if frequency is None:
            frequency = _candidate_frequency(
                index_bytes,
                candidate,
                forbidden_start=anchor_window_start,
                forbidden_end=anchor_window_end,
            )
            frequency_cache[candidate] = frequency
        score = (
            float(base_score)
            + float(anchor_size)
            + (boundary_matches * 8.0)
            + math.log1p(max(0, frequency)) * 3.0
            - (distance / max(256.0, float(len(index_bytes))))
        )
        existing = rows.get(candidate)
        source = {
            "anchor_mode": mode,
            "source_offset": int(candidate_start),
            "source_window_start": int(source_window_start),
            "source_window_end": int(source_window_end),
            "source_window_len": int(source_window_end - source_window_start),
            "anchor_size": int(anchor_size),
            "distance_from_target": int(distance),
            "boundary_matches": int(boundary_matches),
            "overlaps_target": False,
            "overlaps_anchor_window": False,
            "sentinel_in_source_window": False,
        }
        if existing is None:
            rows[candidate] = {
                "ids": _ids_for_bytes(tokenizer, candidate),
                "predicted_hole": _decode_candidate(tokenizer, candidate),
                "source": "in_document_echo",
                "anchor_mode": mode,
                "rank": 0,
                "echo_score": score,
                "support": 1,
                "frequency_in_visible_doc": int(frequency),
                "best_anchor_size": int(anchor_size),
                "min_distance_from_target": int(distance),
                "shape": _shape(candidate),
                "sources": [source],
                "overlaps_target": False,
                "overlaps_anchor_window": False,
                "sentinel_in_source_window": False,
            }
            return
        existing["support"] = int(existing["support"]) + 1
        existing["frequency_in_visible_doc"] = max(int(existing["frequency_in_visible_doc"]), int(frequency))
        existing["best_anchor_size"] = max(int(existing["best_anchor_size"]), int(anchor_size))
        existing["min_distance_from_target"] = min(int(existing["min_distance_from_target"]), int(distance))
        existing["echo_score"] = max(float(existing["echo_score"]), score)
        if len(existing["sources"]) < 12:
            existing["sources"].append(source)

    for raw_size in sizes:
        size = max(1, int(raw_size))
        left_anchor = target_left[-min(size, len(target_left)) :]
        right_anchor = target_right[: min(size, len(target_right))]
        if left_anchor and right_anchor:
            search_at = 0
            while True:
                left_at = index_bytes.find(left_anchor, search_at)
                if left_at < 0:
                    break
                gap_start = left_at + len(left_anchor)
                gap_end = gap_start + hole_len
                window_end = gap_end + len(right_anchor)
                if window_end <= len(index_bytes) and index_bytes[gap_end:window_end] == right_anchor:
                    add_candidate(
                        mode="exact_bi_anchor_echo",
                        candidate_start=gap_start,
                        candidate_end=gap_end,
                        source_window_start=left_at,
                        source_window_end=window_end,
                        anchor_size=min(len(left_anchor), len(right_anchor)),
                        base_score=1000.0,
                    )
                search_at = left_at + 1
        if left_anchor:
            search_at = 0
            while True:
                left_at = index_bytes.find(left_anchor, search_at)
                if left_at < 0:
                    break
                gap_start = left_at + len(left_anchor)
                gap_end = gap_start + hole_len
                add_candidate(
                    mode="left_echo",
                    candidate_start=gap_start,
                    candidate_end=gap_end,
                    source_window_start=left_at,
                    source_window_end=gap_end,
                    anchor_size=len(left_anchor),
                    base_score=500.0,
                )
                search_at = left_at + 1
        if right_anchor:
            search_at = 0
            while True:
                right_at = index_bytes.find(right_anchor, search_at)
                if right_at < 0:
                    break
                gap_start = right_at - hole_len
                gap_end = right_at
                add_candidate(
                    mode="right_echo",
                    candidate_start=gap_start,
                    candidate_end=gap_end,
                    source_window_start=gap_start,
                    source_window_end=right_at + len(right_anchor),
                    anchor_size=len(right_anchor),
                    base_score=500.0,
                )
                search_at = right_at + 1

    span_counts: Counter[bytes] = Counter()
    for start in range(0, max(0, len(index_bytes) - hole_len + 1)):
        end = start + hole_len
        window_start = max(0, start - 1)
        window_end = min(len(index_bytes), end + 1)
        if _overlaps(window_start, window_end, anchor_window_start, anchor_window_end):
            continue
        candidate = index_bytes[start:end]
        if SENTINEL_BYTE in candidate or not candidate.strip():
            continue
        left_neighbor = index_bytes[start - 1] if start > 0 else None
        right_neighbor = index_bytes[end] if end < len(index_bytes) else None
        if _byte_class(left_neighbor) != left_boundary_class and _byte_class(right_neighbor) != right_boundary_class:
            continue
        span_counts[candidate] += 1
    for candidate, count in span_counts.most_common(max(limit * 4, limit)):
        start = index_bytes.find(candidate)
        if start < 0:
            continue
        while start >= 0 and _overlaps(max(0, start - 1), min(len(index_bytes), start + hole_len + 1), anchor_window_start, anchor_window_end):
            start = index_bytes.find(candidate, start + 1)
        if start < 0:
            continue
        add_candidate(
            mode="morphology_echo",
            candidate_start=start,
            candidate_end=start + hole_len,
            source_window_start=max(0, start - 1),
            source_window_end=min(len(index_bytes), start + hole_len + 1),
            anchor_size=0,
            base_score=100.0 + math.log1p(count) * 8.0,
            frequency_override=int(count),
        )

    ranked = sorted(
        rows.values(),
        key=lambda row: (
            -float(row["echo_score"]),
            int(row["min_distance_from_target"]),
            str(row["predicted_hole"]),
        ),
    )[: max(0, int(limit))]
    for rank, row in enumerate(ranked):
        row["rank"] = int(rank)
    return ranked


def echo_overlap_audit(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "target_window_overlap_hits": sum(1 for row in candidates if row.get("overlaps_target")),
        "target_anchor_window_overlap_hits": sum(1 for row in candidates if row.get("overlaps_anchor_window")),
        "same_offset_hits": sum(1 for row in candidates if int(row.get("min_distance_from_target", 0)) == 0),
        "sentinel_source_window_hits": sum(1 for row in candidates if row.get("sentinel_in_source_window")),
    }
