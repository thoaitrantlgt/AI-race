param(
    [string]$InputDir = "data/input",
    [string]$OutputZip = "output.zip",
    [string]$Config = "configs/llm_pseudo_hybrid.yaml"
)

python -m src.main `
  --input_dir "$InputDir" `
  --output_zip "$OutputZip" `
  --config "$Config"
