# Viettel Medical IE Pipeline

Hybrid Vietnamese medical information extraction using Qwen3-8B, rules,
terminology linking, and optional VietMed-NER checkpoints.

## Architecture

```text
Qwen3 entity/type/assertion -------+
Rules and terminology -------------+--> span merge --> assertion repair
Optional VietMed-NER boundaries ---+                         |
                                                             v
                                                  ICD-10/RxNorm linker
```

Qwen3 never supplies ICD-10 or RxNorm identifiers. It extracts entity text,
type, and assertion. The pipeline validates offsets against the original text,
then maps diagnoses and drugs through local terminology data.

The default active-model budget is below the competition limit:

```text
Qwen3-8B                 8.200B
XLM-R VietMed-NER        0.278B (only in the full ensemble profile)
Total                    8.478B / 9.000B
```

## 1. Install Pipeline Dependencies

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `llm_rule` profile does not load Qwen weights in this process. Qwen runs on
an external OpenAI-compatible server, so no LLM package or checkpoint download
is performed by `run_inference.sh`.

## 2. Start the Qwen3 Server

The pipeline expects an OpenAI-compatible API serving model ID
`Qwen/Qwen3-8B`. Use an existing server, or start one with vLLM:

```bash
python -m pip install vllm
vllm serve Qwen/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000
```

Example with llama.cpp and the official GGUF repository:

```bash
llama-server \
  -hf Qwen/Qwen3-8B-GGUF:Q4_K_M \
  --alias Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  -c 16384
```

These server commands may download Qwen when the server does not already have
the checkpoint. The inference script itself never downloads it.

Check the server before running the pipeline:

```bash
curl http://127.0.0.1:8000/v1/models
```

For a remote or authenticated server, edit `models.llm.base_url` in the chosen
config and export the API key:

```bash
export LLM_API_KEY="your-key"
```

PowerShell:

```powershell
$env:LLM_API_KEY = "your-key"
```

Requests set `chat_template_kwargs.enable_thinking=false`, temperature `0`, and
JSON response format for deterministic extraction.

## 3. Prepare Input

Put one medical record in each UTF-8 `.txt` file:

```text
data/input/1.txt
data/input/2.txt
data/input/3.txt
```

File stems become record IDs. Numeric names are processed in numeric order.

## 4. Prepare Terminology

Terminology is optional for entity extraction but required for useful ICD-10
and RxNorm candidates.

```text
data/terminology/icd10_vi/
data/terminology/rxnorm_2026/
data/terminology/custom_aliases/
```

For full RxNorm, place `RXNCONSO.RRF` at:

```text
data/terminology/rxnorm_2026/RXNCONSO.RRF
```

Vietnamese ICD-10 TSV/CSV/JSON files belong under:

```text
data/terminology/icd10_vi/
```

The repository intentionally does not contain competition input, terminology
dumps, model checkpoints, or generated outputs.

## 5. Choose a Profile

### Portable fallback profile: Qwen3 + rules

This profile only requires the Qwen server and optional terminology files. It
does not require local NER or pseudo-model checkpoints. It is useful for smoke
tests and unseen data, but leaderboard testing showed that it is substantially
weaker than the pseudo-model hybrid on the current 100-record set.

```text
configs/llm_rule.yaml
```

### Recommended competition profile: Qwen3 + pseudo-model + rules

This profile restores the high-scoring local extraction baseline and uses
Qwen3 in exact-span consensus mode. Qwen output cannot add an entity unless its
offset and type match a local source.

```text
configs/llm_pseudo_hybrid.yaml
```

Required local artifacts:

```text
data/models/pseudo_ner_model.json
data/models/assertion_classifier.pkl
```

### Full ensemble profile

`configs/default.yaml` additionally loads local pseudo-model, assertion, and
XLM-R VietMed-NER artifacts when available. Place them at:

```text
data/models/pseudo_ner_model.json
data/models/assertion_classifier.pkl
data/models/xlm-roberta-base-VietMed-NER/
```

Install the optional local NER runtime:

```bash
python -m pip install torch transformers
```

Missing local checkpoints are skipped with warnings; they are never downloaded
because `local_files_only` is enabled.

## 6. Run End-to-End Inference

Portable server-only fallback:

```bash
bash run_inference.sh data/input output.zip configs/llm_rule.yaml
```

Recommended competition hybrid:

```bash
bash run_inference.sh data/input output.zip configs/llm_pseudo_hybrid.yaml
```

Full ensemble with local checkpoints:

```bash
bash run_inference.sh data/input output.zip configs/default.yaml
```

Windows PowerShell equivalents:

```powershell
.\run_inference.ps1 -InputDir data/input -OutputZip output.zip -Config configs/llm_rule.yaml
```

```powershell
.\run_inference.ps1 -InputDir data/input -OutputZip output.zip -Config configs/llm_pseudo_hybrid.yaml
```

```powershell
.\run_inference.ps1 -InputDir data/input -OutputZip output.zip -Config configs/default.yaml
```

At startup the pipeline prints the active parameter total. Expected values are:

```text
llm_rule.yaml: 8.200B / 9.000B
llm_pseudo_hybrid.yaml: 8.200B / 9.000B
default.yaml:  8.478B / 9.000B
```

If the LLM endpoint fails, the client logs one warning, disables LLM calls for
the remainder of that run, and continues with available local rules/models.

## 7. Validate Output

```bash
python -m src.io.validate_output output.zip \
  --input_dir data/input \
  --config configs/llm_rule.yaml
```

Successful validation exits with code `0`. The run creates:

```text
output/       # one JSON file per input record
output.zip    # submission archive
```

Each JSON entry has exactly this shape:

```json
{
  "text": "metoprolol 25mg po bid",
  "position": [76, 98],
  "type": "THUỐC",
  "assertions": ["isHistorical"],
  "candidates": ["6918"]
}
```

`position` uses zero-based Python-style spans `[start, end)`, with an exclusive
end offset.

## Troubleshooting

### `Connection refused` or LLM disabled after server error

Verify the endpoint and model alias:

```bash
curl http://127.0.0.1:8000/v1/models
```

The returned model ID must match `models.llm.model` in the config.

### XLM-R checkpoint warning

Use `configs/llm_rule.yaml`, or place the model under
`data/models/xlm-roberta-base-VietMed-NER/`.

### Empty candidate lists

Entity extraction can work without terminology, but candidates require ICD-10
and RxNorm files under `data/terminology/`.

### Parameter-budget error

Only enabled models count. Disable another local model before adding one; the
pipeline refuses to run when the configured total exceeds 9B.
