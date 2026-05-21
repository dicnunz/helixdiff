from __future__ import annotations

from statistics import median
from typing import Any

from ..tokenizer import ByteTokenizer
from .in_document_echo import SENTINEL_BYTE, redacted_document_bytes


ECHO_SUPPORT_MODES = {"exact_bi_anchor_echo", "left_echo", "right_echo", "morphology_echo"}


def _overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def _row_key(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in row.get("ids", []))


def _candidate_bytes(tokenizer: ByteTokenizer, row: dict[str, Any]) -> bytes:
    raw = bytearray()
    for token_id in row.get("ids", []):
        token_id = int(token_id)
        if token_id < tokenizer.byte_offset:
            return str(row.get("predicted_hole", "")).encode("utf-8", errors="replace")
        raw.append(token_id - tokenizer.byte_offset)
    return bytes(raw)


def _source_modes(row: dict[str, Any]) -> set[str]:
    modes = {str(row.get("anchor_mode", ""))}
    modes.update(str(source.get("anchor_mode", "")) for source in row.get("sources", []))
    return {mode for mode in modes if mode in ECHO_SUPPORT_MODES}


def _has_legal_echo_source(row: dict[str, Any]) -> bool:
    if str(row.get("source", "")) == "in_document_echo":
        return True
    return any(str(source.get("source", "")) == "in_document_echo" for source in row.get("sources", []))


def _count_candidate_frequency(
    *,
    index_bytes: bytes,
    candidate: bytes,
    anchor_window_start: int,
    anchor_window_end: int,
) -> int:
    if not candidate:
        return 0
    count = 0
    search_at = 0
    while True:
        found = index_bytes.find(candidate, search_at)
        if found < 0:
            return count
        end = found + len(candidate)
        if (
            not _overlaps(max(0, found - 1), min(len(index_bytes), end + 1), anchor_window_start, anchor_window_end)
            and SENTINEL_BYTE not in index_bytes[max(0, found - 1) : min(len(index_bytes), end + 1)]
        ):
            count += 1
        search_at = found + 1


def _pattern_candidate_offsets(
    *,
    index_bytes: bytes,
    pattern: bytes,
    candidate_offset_in_pattern: int,
    candidate_len: int,
    anchor_window_start: int,
    anchor_window_end: int,
) -> set[int]:
    if not pattern or candidate_len <= 0:
        return set()
    offsets: set[int] = set()
    search_at = 0
    while True:
        found = index_bytes.find(pattern, search_at)
        if found < 0:
            return offsets
        end = found + len(pattern)
        candidate_start = found + candidate_offset_in_pattern
        candidate_end = candidate_start + candidate_len
        if (
            not _overlaps(found, end, anchor_window_start, anchor_window_end)
            and SENTINEL_BYTE not in index_bytes[found:end]
            and candidate_start >= 0
            and candidate_end <= len(index_bytes)
        ):
            offsets.add(candidate_start)
        search_at = found + 1


def echo_candidate_evidence(
    *,
    tokenizer: ByteTokenizer,
    document_text: str,
    span_start: int,
    span_end: int,
    candidate_row: dict[str, Any],
    context_bytes: int,
    anchor_sizes: list[int],
    anchor_window_bytes: int,
    left_context_override: bytes | None = None,
    right_context_override: bytes | None = None,
    include_row_source_modes: bool = True,
) -> dict[str, Any]:
    """Score label-free same-document witness strength for one echo candidate.

    The target span is redacted before evidence is counted. Optional context overrides
    provide contrastive nulls such as blank or swapped anchors without touching labels.
    """

    document_bytes = document_text.encode("utf-8", errors="replace")
    index_bytes = redacted_document_bytes(document_text, span_start, span_end)
    candidate = _candidate_bytes(tokenizer, candidate_row)
    anchor_window_start = max(0, int(span_start) - max(0, int(anchor_window_bytes)))
    anchor_window_end = min(len(index_bytes), int(span_end) + max(0, int(anchor_window_bytes)))
    left_context = (
        left_context_override
        if left_context_override is not None
        else document_bytes[max(0, int(span_start) - int(context_bytes)) : int(span_start)]
    )
    right_context = (
        right_context_override
        if right_context_override is not None
        else document_bytes[int(span_end) : min(len(document_bytes), int(span_end) + int(context_bytes))]
    )

    exact_offsets: set[int] = set()
    left_offsets: set[int] = set()
    right_offsets: set[int] = set()
    for raw_size in anchor_sizes:
        size = max(1, int(raw_size))
        left_anchor = left_context[-min(size, len(left_context)) :]
        right_anchor = right_context[: min(size, len(right_context))]
        if left_anchor and right_anchor:
            exact_offsets.update(
                _pattern_candidate_offsets(
                    index_bytes=index_bytes,
                    pattern=left_anchor + candidate + right_anchor,
                    candidate_offset_in_pattern=len(left_anchor),
                    candidate_len=len(candidate),
                    anchor_window_start=anchor_window_start,
                    anchor_window_end=anchor_window_end,
                )
            )
        if left_anchor:
            left_offsets.update(
                _pattern_candidate_offsets(
                    index_bytes=index_bytes,
                    pattern=left_anchor + candidate,
                    candidate_offset_in_pattern=len(left_anchor),
                    candidate_len=len(candidate),
                    anchor_window_start=anchor_window_start,
                    anchor_window_end=anchor_window_end,
                )
            )
        if right_anchor:
            right_offsets.update(
                _pattern_candidate_offsets(
                    index_bytes=index_bytes,
                    pattern=candidate + right_anchor,
                    candidate_offset_in_pattern=0,
                    candidate_len=len(candidate),
                    anchor_window_start=anchor_window_start,
                    anchor_window_end=anchor_window_end,
                )
            )

    source_modes = _source_modes(candidate_row) if include_row_source_modes else set()
    morphology_count = (
        max(1, int(candidate_row.get("frequency_in_visible_doc", 0)))
        if include_row_source_modes and "morphology_echo" in source_modes
        else 0
    )
    mode_count = sum(
        1
        for count in (
            len(exact_offsets),
            len(left_offsets),
            len(right_offsets),
            morphology_count,
        )
        if count > 0
    )
    source_offsets = exact_offsets | left_offsets | right_offsets
    frequency = _count_candidate_frequency(
        index_bytes=index_bytes,
        candidate=candidate,
        anchor_window_start=anchor_window_start,
        anchor_window_end=anchor_window_end,
    )
    evidence = {
        "exact_bi_anchor_witness_count": len(exact_offsets),
        "left_echo_witness_count": len(left_offsets),
        "right_echo_witness_count": len(right_offsets),
        "morphology_echo_witness_count": int(morphology_count),
        "witness_source_count": len(source_offsets),
        "witness_mode_count": int(mode_count),
        "candidate_visible_doc_frequency": int(frequency),
        "row_source_modes": sorted(source_modes),
        "forbidden_overlap_flags": {
            "overlaps_target": bool(candidate_row.get("overlaps_target")),
            "overlaps_anchor_window": bool(candidate_row.get("overlaps_anchor_window")),
            "sentinel_in_source_window": bool(candidate_row.get("sentinel_in_source_window")),
        },
    }
    evidence["score"] = echo_dominance_score(evidence)
    return evidence


def echo_dominance_score(evidence: dict[str, Any]) -> float:
    return float(
        8 * int(evidence.get("exact_bi_anchor_witness_count", 0))
        + 3
        * min(
            int(evidence.get("left_echo_witness_count", 0)),
            int(evidence.get("right_echo_witness_count", 0)),
        )
        + 2 * int(evidence.get("witness_mode_count", 0))
        + int(evidence.get("witness_source_count", 0))
        + int(evidence.get("morphology_echo_witness_count", 0))
    )


def _evidence_is_promotable(evidence: dict[str, Any], *, min_real_null_margin: float) -> bool:
    return float(evidence.get("score", 0.0)) >= float(min_real_null_margin) and (
        int(evidence.get("exact_bi_anchor_witness_count", 0)) > 0
        or int(evidence.get("witness_mode_count", 0)) >= 2
    )


def echo_dominance_decision(
    *,
    tokenizer: ByteTokenizer,
    document_text: str,
    span_start: int,
    span_end: int,
    candidate_row: dict[str, Any],
    context_bytes: int,
    anchor_sizes: list[int],
    anchor_window_bytes: int,
    null_contexts: list[dict[str, bytes]],
    min_real_null_margin: float,
) -> dict[str, Any]:
    real = echo_candidate_evidence(
        tokenizer=tokenizer,
        document_text=document_text,
        span_start=span_start,
        span_end=span_end,
        candidate_row=candidate_row,
        context_bytes=context_bytes,
        anchor_sizes=anchor_sizes,
        anchor_window_bytes=anchor_window_bytes,
    )
    nulls = []
    for null_context in null_contexts:
        null_evidence = echo_candidate_evidence(
            tokenizer=tokenizer,
            document_text=document_text,
            span_start=span_start,
            span_end=span_end,
            candidate_row=candidate_row,
            context_bytes=context_bytes,
            anchor_sizes=anchor_sizes,
            anchor_window_bytes=anchor_window_bytes,
            left_context_override=null_context.get("left", b""),
            right_context_override=null_context.get("right", b""),
            include_row_source_modes=False,
        )
        nulls.append({"name": str(null_context.get("name", "null")), **null_evidence})
    null_score = max((float(item["score"]) for item in nulls), default=0.0)
    flags = real["forbidden_overlap_flags"]
    causal_margin = float(real["score"]) - null_score
    promoted_under_blank = any(
        item["name"] == "blank" and _evidence_is_promotable(item, min_real_null_margin=min_real_null_margin)
        for item in nulls
    )
    promoted_under_swapped = any(
        item["name"] == "swapped_edges" and _evidence_is_promotable(item, min_real_null_margin=min_real_null_margin)
        for item in nulls
    )
    promotable = (
        _has_legal_echo_source(candidate_row)
        and not any(bool(value) for value in flags.values())
        and causal_margin >= float(min_real_null_margin)
        and _evidence_is_promotable(real, min_real_null_margin=min_real_null_margin)
        and not promoted_under_blank
        and not promoted_under_swapped
    )
    return {
        "candidate_key": list(_row_key(candidate_row)),
        "predicted_hole": str(candidate_row.get("predicted_hole", "")),
        "real": real,
        "nulls": nulls,
        "null_score": null_score,
        "causal_margin": causal_margin,
        "promotable": promotable,
        "promoted_under_blank": promoted_under_blank,
        "promoted_under_swapped_edges": promoted_under_swapped,
    }


def _dedupe_ranked(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[int, ...]] = set()
    ranked: list[dict[str, Any]] = []
    for row in rows:
        key = _row_key(row)
        if key in seen:
            continue
        copied = dict(row)
        copied["rank"] = len(ranked)
        ranked.append(copied)
        seen.add(key)
        if len(ranked) >= limit:
            break
    return ranked


def rank_with_echo_dominance(
    *,
    tokenizer: ByteTokenizer,
    document_text: str,
    span_start: int,
    span_end: int,
    prior: list[dict[str, Any]],
    echo: list[dict[str, Any]],
    context_bytes: int,
    anchor_sizes: list[int],
    anchor_window_bytes: int,
    null_contexts: list[dict[str, bytes]],
    min_real_null_margin: float = 2.0,
    max_rank_promotions: int = 1,
    max_candidates: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decisions = [
        echo_dominance_decision(
            tokenizer=tokenizer,
            document_text=document_text,
            span_start=span_start,
            span_end=span_end,
            candidate_row=row,
            context_bytes=context_bytes,
            anchor_sizes=anchor_sizes,
            anchor_window_bytes=anchor_window_bytes,
            null_contexts=null_contexts,
            min_real_null_margin=min_real_null_margin,
        )
        for row in echo
    ]
    decision_by_key = {tuple(item["candidate_key"]): item for item in decisions}
    promotable = [
        row
        for row in echo
        if bool(decision_by_key.get(_row_key(row), {}).get("promotable"))
    ]
    promotable.sort(
        key=lambda row: (
            -float(decision_by_key[_row_key(row)]["causal_margin"]),
            -float(decision_by_key[_row_key(row)]["real"]["score"]),
            -float(row.get("echo_score", 0.0)),
            int(row.get("rank", 0)),
        )
    )
    promotions = promotable[: max(0, int(max_rank_promotions))]
    ranked = _dedupe_ranked([*promotions, *prior, *echo], limit=max_candidates)
    promoted_keys = {_row_key(row) for row in promotions}
    promoted_decisions = [decision_by_key[key] for key in promoted_keys if key in decision_by_key]
    real_scores = [float(item["real"]["score"]) for item in decisions]
    null_scores = [float(item["null_score"]) for item in decisions]
    margins = [float(item["causal_margin"]) for item in decisions]
    summary = {
        "promoted_cases": int(bool(promotions)),
        "rank1_promotions": int(bool(promotions)),
        "rank2_to_rank4_promotions": max(0, min(3, len(promotions) - 1)),
        "promoted_candidate_count": len(promotions),
        "promoted_candidate_keys": [list(_row_key(row)) for row in promotions],
        "median_real_score": float(median(real_scores)) if real_scores else 0.0,
        "median_null_score": float(median(null_scores)) if null_scores else 0.0,
        "median_causal_margin": float(median(margins)) if margins else 0.0,
        "promoted_real_score_median": float(median([float(item["real"]["score"]) for item in promoted_decisions]))
        if promoted_decisions
        else 0.0,
        "promoted_null_score_max": max((float(item["null_score"]) for item in promoted_decisions), default=0.0),
        "promoted_causal_margin_min": min((float(item["causal_margin"]) for item in promoted_decisions), default=0.0),
        "promoted_under_blank": sum(1 for item in decisions if bool(item["promoted_under_blank"])),
        "promoted_under_swapped_edges": sum(1 for item in decisions if bool(item["promoted_under_swapped_edges"])),
        "decisions": decisions,
    }
    return ranked, summary
