from src.linking.alias_loader import load_custom_aliases
from src.linking.candidate_generator import generate_candidates
from src.linking.candidate_ranker import rank_candidates
from src.linking.rxnorm_loader import load_rxnorm_to_store
from src.linking.terminology_store import TermEntry, TerminologyStore


def test_builtin_alias_linker():
    store = TerminologyStore()
    load_custom_aliases("missing", store)
    candidates = generate_candidates("tylenol", "drug", "", store)
    assert any(candidate.concept_id == "Rx161" for candidate in candidates)


def test_rxnorm_loader_keeps_strength_components_and_deduplicates(tmp_path):
    root = tmp_path / "rxnorm_2026"
    root.mkdir()

    def row(rxcui: str, tty: str, name: str) -> str:
        fields = [rxcui, "ENG", "", "", "", "", "", "", "", "", "", "RXNORM", tty, rxcui, name, "", "N", ""]
        return "|".join(fields) + "|\n"

    (root / "RXNCONSO.RRF").write_text(
        row("315643", "SCDC", "chlorpheniramine 0.4 MG/ML")
        + row("315643", "SCDC", "chlorpheniramine 0.4 MG/ML")
        + row("315643", "SY", "chlorpheniramine strength synonym"),
        encoding="utf-8",
    )
    store = TerminologyStore()
    load_rxnorm_to_store(root, store)

    matches = store.lookup_exact("chlorpheniramine 0.4 mg/ml")
    assert [(entry.concept_id, entry.version) for entry in matches] == [("Rx315643", "2026-07-06")]


def test_custom_alias_loader_can_skip_redundant_rxnav_bulk(tmp_path):
    (tmp_path / "rxnorm_rxnav_2026.tsv").write_text(
        "bulk drug\tbulk drug\tRx100\tRxNorm\tdrug\n", encoding="utf-8"
    )
    (tmp_path / "rxnorm_seed.tsv").write_text(
        "seed drug\tseed drug\tRx200\tRxNorm\tdrug\n", encoding="utf-8"
    )
    store = TerminologyStore()
    load_custom_aliases(tmp_path, store, skip_rxnorm_bulk=True)

    assert not store.lookup_exact("bulk drug")
    assert store.lookup_exact("seed drug")[0].concept_id == "Rx200"


def add_drug(store: TerminologyStore, concept_id: str, name: str, term_type: str) -> None:
    store.add_entry(TermEntry(concept_id, name, "", "RxNorm", "drug", term_type=term_type))


def top_drug_candidate(store: TerminologyStore, mention: str) -> str:
    generated = generate_candidates(mention, "drug", "", store)
    return rank_candidates(mention, "drug", generated, "")[0].concept_id


def test_strength_linking_prefers_clinical_drug_over_ingredient():
    store = TerminologyStore()
    add_drug(store, "Rx17767", "amlodipine", "IN")
    add_drug(store, "Rx308135", "amlodipine 10 MG Oral Tablet", "SCD")

    assert top_drug_candidate(store, "amlodipine 10 mg po daily") == "Rx308135"


def test_strength_linking_prefers_lower_endpoint_for_dose_range():
    store = TerminologyStore()
    add_drug(store, "Rx313782", "acetaminophen 325 MG Oral Tablet", "SCD")
    add_drug(store, "Rx198444", "acetaminophen 650 MG Oral Tablet", "SCD")

    assert top_drug_candidate(store, "acetaminophen 325-650 mg po q6h:prn") == "Rx313782"


def test_administration_volume_does_not_match_another_ingredient():
    store = TerminologyStore()
    add_drug(store, "Rx373136", "nystatin Oral Suspension", "SCDF")
    add_drug(store, "Rx197803", "ibuprofen 100 MG in 5 ML Oral Suspension", "SCD")

    assert top_drug_candidate(store, "nystatin oral suspension 5 ml po qid:prn") == "Rx373136"
