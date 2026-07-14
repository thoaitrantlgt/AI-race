from __future__ import annotations

from dataclasses import dataclass, field
import re

from src.preprocess.text_normalizer import fold_for_matching, normalize_for_matching


@dataclass
class TermEntry:
    concept_id: str
    name: str
    normalized_name: str
    source: str
    semantic_type: str | None
    aliases: list[str] = field(default_factory=list)
    version: str | None = None
    language: str | None = None
    term_type: str | None = None


DRUG_STRENGTH_RE = re.compile(
    r"(?<![\d.,])(\d+(?:[.,]\d+)?)[ \t]*(mg|mcg|µg|g|ml|iu|unit|units|%)(?![A-Za-z])",
    re.IGNORECASE,
)
DRUG_STRENGTH_RANGE_RE = re.compile(
    r"(?<![\d.,])(\d+(?:[.,]\d+)?)[ \t]*-[ \t]*(\d+(?:[.,]\d+)?)[ \t]*"
    r"(mg|mcg|µg|g|ml|iu|unit|units|%)(?![A-Za-z])",
    re.IGNORECASE,
)


def drug_strength_keys(text: str) -> list[str]:
    keys: list[str] = []
    for low, high, unit in DRUG_STRENGTH_RANGE_RE.findall(text):
        for value in (low, high):
            key = f"{value.replace(',', '.')} {unit.lower()}"
            if key not in keys:
                keys.append(key)
    for value, unit in DRUG_STRENGTH_RE.findall(text):
        key = f"{value.replace(',', '.')} {unit.lower()}"
        if key not in keys:
            keys.append(key)
    return keys


class TerminologyStore:
    def __init__(self):
        self.entries: list[TermEntry] = []
        self.exact_index: dict[str, list[TermEntry]] = {}
        self.alias_index: dict[str, list[TermEntry]] = {}
        self.folded_index: dict[str, list[TermEntry]] = {}
        self.token_index: dict[str, list[TermEntry]] = {}
        self.extraction_aliases: dict[str, set[str]] = {}
        self.drug_strength_index: dict[str, list[TermEntry]] = {}
        self._fuzzy_choices: list[str] | None = None

    def add_entry(self, entry: TermEntry) -> None:
        if not entry.normalized_name:
            entry.normalized_name = normalize_for_matching(entry.name)
        self.entries.append(entry)
        self.exact_index.setdefault(entry.normalized_name, []).append(entry)
        folded_name = fold_for_matching(entry.name)
        if folded_name and folded_name != entry.normalized_name:
            self.folded_index.setdefault(folded_name, []).append(entry)
        if entry.semantic_type:
            self.extraction_aliases.setdefault(entry.semantic_type, set()).add(entry.name)
        if entry.semantic_type == "drug" and entry.term_type in {
            "SCD",
            "SBD",
            "SCDC",
            "SBDC",
            "SCDF",
            "SBDF",
            "SCDG",
            "SBDG",
            "PSN",
            "CD",
        }:
            for strength in drug_strength_keys(entry.name):
                self.drug_strength_index.setdefault(strength, []).append(entry)
        for alias in entry.aliases:
            norm = normalize_for_matching(alias)
            self.alias_index.setdefault(norm, []).append(entry)
            folded_alias = fold_for_matching(alias)
            if folded_alias and folded_alias != norm:
                self.folded_index.setdefault(folded_alias, []).append(entry)
            if entry.semantic_type:
                self.extraction_aliases.setdefault(entry.semantic_type, set()).add(alias)
        for token in set(entry.normalized_name.split()):
            if len(token) >= 3:
                self.token_index.setdefault(token, []).append(entry)
        self._fuzzy_choices = None

    def add_extraction_alias(self, alias: str, semantic_type: str) -> None:
        self.extraction_aliases.setdefault(semantic_type, set()).add(alias)

    def aliases_by_type(self, semantic_type: str) -> set[str]:
        return self.extraction_aliases.get(semantic_type, set())

    def lookup_exact(self, mention: str) -> list[TermEntry]:
        return self.exact_index.get(normalize_for_matching(mention), [])

    def lookup_alias(self, mention: str) -> list[TermEntry]:
        return self.alias_index.get(normalize_for_matching(mention), [])

    def lookup_folded_exact(self, mention: str) -> list[TermEntry]:
        folded = fold_for_matching(mention)
        return self.folded_index.get(folded, [])

    def lookup_by_tokens(self, mention: str, limit: int = 20) -> list[TermEntry]:
        seen: dict[str, TermEntry] = {}
        for token in normalize_for_matching(mention).split():
            for entry in self.token_index.get(token, []):
                seen.setdefault(entry.concept_id, entry)
                if len(seen) >= limit:
                    return list(seen.values())
        return list(seen.values())

    def lookup_drug_strength(self, mention: str) -> list[TermEntry]:
        seen: dict[tuple[str, str], TermEntry] = {}
        for strength in drug_strength_keys(mention):
            for entry in self.drug_strength_index.get(strength, []):
                seen.setdefault((entry.concept_id, entry.normalized_name), entry)
        return list(seen.values())

    def lookup_fuzzy(self, mention: str, limit: int = 20) -> list[TermEntry]:
        norm = normalize_for_matching(mention)
        try:
            from rapidfuzz import fuzz, process
        except Exception:
            return self.lookup_by_tokens(mention, limit)
        if self._fuzzy_choices is None:
            self._fuzzy_choices = list(self.exact_index.keys())
        matches = process.extract(norm, self._fuzzy_choices, scorer=fuzz.token_set_ratio, score_cutoff=88, limit=limit)
        entries: list[TermEntry] = []
        seen: set[str] = set()
        for choice, _score, _idx in matches:
            for entry in self.exact_index.get(choice, []):
                if entry.concept_id not in seen:
                    seen.add(entry.concept_id)
                    entries.append(entry)
                    if len(entries) >= limit:
                        return entries
        return entries
