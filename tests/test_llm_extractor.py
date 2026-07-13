from src.assertion.assertion_classifier import AssertionClassifier
from src.extraction.llm_extractor import LlmExtractor
from src.io.read_input import InputRecord
from src.main import filter_llm_spans, validate_parameter_budget


def test_llm_extractor_parses_and_aligns_exact_mentions():
    extractor = LlmExtractor({"enabled": True})
    extractor._request = lambda _: """{
      "entities": [
        {"text":"không ho", "position":[14,22], "type":"TRIỆU_CHỨNG", "assertions":["isNegated"]},
        {"text":"aspirin", "position":[28,35], "type":"THUỐC", "assertions":[]}
      ]
    }"""
    record = InputRecord("1", "1.txt", "Bệnh nhân nói không ho và dùng aspirin.")
    spans = extractor.extract(record, [])
    assert [(span.text, span.concept_type) for span in spans] == [("không ho", "symptom"), ("aspirin", "drug")]
    assert spans[0].assertions == ["isNegated"]


def test_llm_extractor_ignores_nested_assertion_dicts():
    extractor = LlmExtractor({"enabled": True})
    extractor._request = lambda _: """{
      "entities": [
        {"text":"ho", "position":[0,2], "type":"TRIỆU_CHỨNG", "assertions":[{"value":"isNegated"}, {"name":"isFamily"}, "isHistorical", {"bad": "value"}]}
      ]
    }"""
    record = InputRecord("1", "1.txt", "ho")
    spans = extractor.extract(record, [])
    assert spans[0].assertions == ["isNegated", "isFamily", "isHistorical"]


def test_llm_extractor_accepts_boolean_assertion_objects():
    extractor = LlmExtractor({"enabled": True})
    extractor._request = lambda _: """{
      "entities": [
        {"text":"ho", "position":[0,2], "type":"TRIỆU_CHỨNG", "assertions":{"isNegated":true,"isFamily":false}}
      ]
    }"""
    spans = extractor.extract(InputRecord("1", "1.txt", "ho"), [])
    assert spans[0].assertions == ["isNegated"]


def test_assertion_classifier_handles_invalid_pickle(tmp_path):
    bad_model = tmp_path / "bad.pkl"
    bad_model.write_bytes(b"not a valid pickle")
    classifier = AssertionClassifier(str(bad_model))
    assert classifier.enabled is False
    assert classifier.sklearn_model is None


def test_parameter_budget_accepts_qwen3_and_xlmr():
    config = {
        "models": {
            "parameter_budget_billion": 9.0,
            "llm": {"enabled": True, "active_parameters_billion": 8.2},
            "transformer_ner_models": [{"enabled": True, "active_parameters_billion": 0.278}],
        }
    }
    assert validate_parameter_budget(config) == 8.478


def test_llm_consensus_rejects_unverified_entities():
    extractor = LlmExtractor({"enabled": True})
    extractor._request = lambda _: """{
      "entities": [
        {"text":"ho", "position":[0,2], "type":"TRIỆU_CHỨNG", "assertions":[]},
        {"text":"sốt", "position":[6,9], "type":"TRIỆU_CHỨNG", "assertions":[]}
      ]
    }"""
    record = InputRecord("1", "1.txt", "ho và sốt")
    llm_spans = extractor.extract(record, [])
    local_spans = [llm_spans[0]]
    assert filter_llm_spans(llm_spans, local_spans, "consensus") == [llm_spans[0]]
