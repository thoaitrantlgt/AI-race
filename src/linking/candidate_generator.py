from __future__ import annotations

import re
from dataclasses import dataclass

from src.linking.terminology_store import TermEntry, TerminologyStore, drug_strength_keys
from src.preprocess.text_normalizer import normalize_for_matching


@dataclass
class CandidateMatch:
    concept_id: str
    name: str
    source: str
    score: float
    match_type: str
    semantic_type: str | None = None


def _compatible(entry: TermEntry, concept_type: str) -> bool:
    if concept_type == "drug":
        return entry.semantic_type == "drug" or entry.source == "RxNorm"
    if concept_type in {"diagnosis", "symptom"}:
        return entry.semantic_type in {"diagnosis", "problem", "symptom", None} and entry.source != "RxNorm"
    return True


def _base_mentions(mention: str, concept_type: str) -> list[str]:
    mentions = [mention]
    if concept_type == "drug":
        stripped = re.split(r"\s+\d", mention, maxsplit=1)[0]
        if stripped and stripped != mention:
            mentions.append(stripped)
    return mentions


DRUG_SIG_TOKENS = {
    "po",
    "iv",
    "im",
    "sc",
    "daily",
    "bid",
    "tid",
    "qid",
    "qhs",
    "qam",
    "q6h",
    "prn",
    "xl",
    "xr",
    "er",
    "sr",
    "cr",
}
CLINICAL_TTY_BONUS = {
    "SCD": 0.07,
    "SBD": 0.06,
    "PSN": 0.06,
    "SCDC": 0.03,
    "SBDC": 0.02,
    "SCDF": 0.02,
    "SBDF": 0.01,
    "SCDG": 0.01,
    "SBDG": 0.01,
    "CD": 0.065,
}
DRUG_FORM_TERMS = {
    "tablet",
    "capsule",
    "suspension",
    "solution",
    "cream",
    "ointment",
    "injection",
    "patch",
    "suppository",
    "powder",
    "granules",
    "spray",
}


def _drug_core_tokens(text: str) -> set[str]:
    before_strength = re.split(r"\s+\d", normalize_for_matching(text), maxsplit=1)[0]
    return {
        token
        for token in re.findall(r"[a-z]+", before_strength)
        if len(token) > 1 and token not in DRUG_SIG_TOKENS
    }


def _drug_primary_token(text: str) -> str:
    before_strength = re.split(r"\s+\d", normalize_for_matching(text), maxsplit=1)[0]
    for token in re.findall(r"[a-z]+", before_strength):
        if len(token) > 1 and token not in DRUG_SIG_TOKENS and token not in DRUG_FORM_TERMS and token != "oral":
            return token
    return ""


def _drug_form_score(mention_text: str, candidate_name: str) -> float:
    mention = normalize_for_matching(mention_text)
    candidate = normalize_for_matching(candidate_name)
    mentioned_forms = {form for form in DRUG_FORM_TERMS if form in mention}
    candidate_forms = {form for form in DRUG_FORM_TERMS if form in candidate}
    score = 0.0
    if mentioned_forms:
        score += 0.04 if mentioned_forms & candidate_forms else -0.08
    elif re.search(r"(?<!\w)po(?!\w)", mention):
        if "oral tablet" in candidate:
            score += 0.02
        elif "oral" not in candidate:
            score -= 0.03
    extended_requested = any(re.search(rf"(?<!\w){marker}(?!\w)", mention) for marker in ("xl", "xr", "er", "sr"))
    if extended_requested:
        score += 0.03 if "extended release" in candidate else -0.06
    if "daily" in mention and ("24 hr" in candidate or "24hr" in candidate):
        score += 0.015
    return score


def _strength_candidates(mention_text: str, store: TerminologyStore) -> list[CandidateMatch]:
    mention_tokens = _drug_core_tokens(mention_text)
    primary_token = _drug_primary_token(mention_text)
    mention_strengths = drug_strength_keys(mention_text)
    if not mention_tokens or not primary_token or not mention_strengths:
        return []
    mention_has_combo = "/" in mention_text or "+" in mention_text
    matches: dict[str, CandidateMatch] = {}
    for entry in store.lookup_drug_strength(mention_text):
        candidate_tokens = set(re.findall(r"[a-z]+", normalize_for_matching(entry.name)))
        if primary_token not in candidate_tokens:
            continue
        overlap = len(mention_tokens & candidate_tokens) / len(mention_tokens)
        if overlap < 0.6:
            continue
        if not mention_has_combo and "/" in entry.name:
            continue
        entry_strengths = drug_strength_keys(entry.name)
        preferred_strength_bonus = 0.01 if mention_strengths[0] in entry_strengths else 0.0
        score = (
            1.0
            + 0.05 * overlap
            + CLINICAL_TTY_BONUS.get(entry.term_type or "", 0.0)
            + _drug_form_score(mention_text, entry.name)
            + preferred_strength_bonus
        )
        candidate = CandidateMatch(
            entry.concept_id,
            entry.name,
            entry.source,
            score,
            "strength",
            entry.semantic_type,
        )
        old = matches.get(entry.concept_id)
        if old is None or candidate.score > old.score:
            matches[entry.concept_id] = candidate
    return list(matches.values())


def generate_candidates(mention_text: str, concept_type: str, context_window: str, store: TerminologyStore) -> list[CandidateMatch]:
    if concept_type not in {"drug", "diagnosis"}:
        return []
    found: dict[str, CandidateMatch] = {}
    if concept_type == "drug":
        for candidate in _strength_candidates(mention_text, store):
            found[candidate.concept_id] = candidate
    for mention in _base_mentions(mention_text, concept_type):
        for entry in store.lookup_exact(mention):
            if _compatible(entry, concept_type):
                found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, 1.0, "exact", entry.semantic_type))
        for entry in store.lookup_alias(mention):
            if _compatible(entry, concept_type):
                found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, 0.95, "alias", entry.semantic_type))
        for entry in store.lookup_folded_exact(mention):
            if _compatible(entry, concept_type):
                found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, 0.92, "folded_exact", entry.semantic_type))
    if found:
        return list(found.values())
    for entry in store.lookup_fuzzy(mention_text, limit=20):
        if _compatible(entry, concept_type):
            score = 0.82
            if normalize_for_matching(mention_text) in entry.normalized_name or entry.normalized_name in normalize_for_matching(mention_text):
                score = 0.88
            found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, score, "fuzzy", entry.semantic_type))
    if found:
        return list(found.values())
    for entry in store.lookup_by_tokens(mention_text, limit=20):
        if _compatible(entry, concept_type):
            found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, 0.70, "token", entry.semantic_type))
    return list(found.values())
