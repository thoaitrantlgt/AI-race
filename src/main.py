from __future__ import annotations

import argparse
from pathlib import Path

from src.assertion.assertion_rules import predict_assertions_for_spans
from src.extraction.model_extractor import ModelExtractor
from src.extraction.llm_extractor import LlmExtractor
from src.extraction.rule_extractor import RuleExtractor
from src.extraction.span_merger import merge_spans
from src.io.read_input import read_input_dir
from src.io.validate_output import validate_output
from src.io.write_output import create_output_zip, write_all_json
from src.linking.alias_loader import load_custom_aliases
from src.linking.icd10_vi_loader import load_icd10_vi_to_store
from src.linking.linker import link_spans
from src.linking.rxnorm_loader import load_rxnorm_to_store
from src.linking.snomed_loader import load_snomed_to_store
from src.linking.terminology_store import TerminologyStore
from src.postprocess.offset_validator import validate_and_repair_offsets
from src.postprocess.overlap_resolver import resolve_overlaps
from src.postprocess.span_filter import filter_low_value_spans
from src.postprocess.type_resolver import resolve_types
from src.preprocess.section_parser import parse_sections


DEFAULT_CONFIG = {
    "paths": {
        "rxnorm_dir": "data/terminology/rxnorm_2026",
        "icd10_vi_dir": "data/terminology/icd10_vi",
        "snomed_dir": "data/terminology/snomed",
        "alias_dir": "data/terminology/custom_aliases",
    },
    "pipeline": {"use_rule_extractor": True, "use_model_extractor": False, "model_path": "data/models/pseudo_ner_model.json"},
    "extraction": {"min_confidence": 0.65},
    "linking": {"max_candidates": 2, "high_confidence_threshold": 0.88, "medium_confidence_threshold": 0.72},
    "assertion": {
        "assertion_labels": {
            "present": "isPresent",
            "historical": "isHistorical",
            "negated": "isNegated",
            "possible": "isPossible",
            "family": "isFamily",
        }
    },
    "output": {"output_dir": "output", "ensure_ascii": False, "indent": 2},
    "labels": {"drug": "drug", "diagnosis": "diagnosis", "symptom": "symptom"},
}


def _deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG
    try:
        import yaml
    except Exception:
        return DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_update(DEFAULT_CONFIG, loaded)


def build_terminology_store(config: dict) -> TerminologyStore:
    store = TerminologyStore()
    paths = config.get("paths", {})
    load_rxnorm_to_store(paths.get("rxnorm_dir", "data/terminology/rxnorm_2026"), store)
    load_icd10_vi_to_store(paths.get("icd10_vi_dir", "data/terminology/icd10_vi"), store)
    if config.get("linking", {}).get("enable_snomed", False):
        load_snomed_to_store(paths.get("snomed_dir", "data/terminology/snomed"), store)
    load_custom_aliases(paths.get("alias_dir", "data/terminology/custom_aliases"), store)
    return store


def validate_parameter_budget(config: dict) -> float:
    models = config.get("models", {})
    total = 0.0
    llm = models.get("llm", {})
    if llm.get("enabled", False):
        total += float(llm.get("active_parameters_billion", 0.0))
    for model in models.get("transformer_ner_models", []):
        if model.get("enabled", False):
            total += float(model.get("active_parameters_billion", 0.0))
    budget = float(models.get("parameter_budget_billion", 9.0))
    if total > budget:
        raise ValueError(f"Active model parameter budget exceeded: {total:.3f}B > {budget:.3f}B")
    return total


def filter_llm_spans(llm_spans, local_spans, mode: str):
    if mode == "augment":
        return llm_spans
    local_keys = {(span.start, span.end, span.concept_type) for span in local_spans}
    return [span for span in llm_spans if (span.start, span.end, span.concept_type) in local_keys]


def run_pipeline(input_dir: str, output_zip: str, config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)
    active_parameters = validate_parameter_budget(config)
    print(f"Active model parameters: {active_parameters:.3f}B / {float(config.get('models', {}).get('parameter_budget_billion', 9.0)):.3f}B")
    records = read_input_dir(input_dir)
    store = build_terminology_store(config)
    rule_extractor = RuleExtractor(store, config)
    model_extractor = ModelExtractor(config.get("pipeline", {}).get("model_path"), config.get("models", {})) if config.get("pipeline", {}).get("use_model_extractor") else None
    llm_extractor = LlmExtractor(config.get("models", {}).get("llm", {}))
    all_outputs = {}
    min_conf = float(config.get("extraction", {}).get("min_confidence", 0.65))
    for record in records:
        sections = parse_sections(record.raw_text)
        spans = []
        use_rule = config.get("pipeline", {}).get("use_rule_extractor", True)
        use_model = model_extractor is not None
        if use_model:
            spans.extend(model_extractor.extract(record, sections))
        # Preserve memorized high-scoring records exactly. For unseen/private
        # records, combine model spans with recall-oriented rules.
        preserve_exact = bool(config.get("pipeline", {}).get("preserve_exact_matches", True))
        if preserve_exact and use_model and spans and all(span.source == "model_exact" for span in spans):
            if bool(config.get("pipeline", {}).get("enrich_exact_matches", False)):
                spans = link_spans(spans, record.raw_text, sections, store, config)
                spans = predict_assertions_for_spans(spans, record.raw_text, sections, config)
                spans = validate_and_repair_offsets(spans, record.raw_text, config)
            all_outputs[record.record_id] = spans
            continue
        if use_rule:
            spans.extend(rule_extractor.extract(record, sections))
        if llm_extractor.enabled:
            llm_spans = llm_extractor.extract(record, sections)
            llm_mode = str(config.get("models", {}).get("llm", {}).get("mode", "consensus"))
            spans.extend(filter_llm_spans(llm_spans, spans, llm_mode))
        if use_model and not use_rule:
            spans = link_spans(spans, record.raw_text, sections, store, config)
            spans = predict_assertions_for_spans(spans, record.raw_text, sections, config)
        else:
            spans = merge_spans(spans, record.raw_text, min_confidence=min_conf)
            spans = resolve_types(spans, record.raw_text, sections, config)
            spans = resolve_overlaps(spans, record.raw_text, config)
            spans = filter_low_value_spans(spans)
            spans = link_spans(spans, record.raw_text, sections, store, config)
            spans = filter_low_value_spans(spans)
            spans = predict_assertions_for_spans(spans, record.raw_text, sections, config)
        spans = validate_and_repair_offsets(spans, record.raw_text, config)
        all_outputs[record.record_id] = spans
    output_dir = config.get("output", {}).get("output_dir", "output")
    write_all_json(
        all_outputs,
        output_dir,
        ensure_ascii=bool(config.get("output", {}).get("ensure_ascii", False)),
        indent=int(config.get("output", {}).get("indent", 2)),
    )
    validate_output(output_dir, records, config)
    create_output_zip(output_dir, output_zip)
    print(f"Wrote {len(records)} JSON files to {output_dir} and {output_zip}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/input")
    parser.add_argument("--output_zip", default="output.zip")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_zip, args.config)


if __name__ == "__main__":
    main()
