# Repository Guidelines

## Project Structure & Module Organization
- `blip3o/`: core model code (model, training, WAN video components).
- `deepspeed_scripts/`: DeepSpeed JSON configs.
- `inference.py`, `run.sh`, `sft.sh`, `slurm.sh`: main entrypoints.
- `gradio/`: demo UI and sample assets.
- `siglip2_sana/`: SigLIP2 + SANA pipeline.
- `eval/`: evaluation scripts; `eval/lmms-eval/` vendored eval framework; `eval/geneval/` generation.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs core dependencies.
- `python inference.py /path/to/checkpoint` runs base inference.
- `bash run.sh` launches local multi-GPU training via torchrun.
- `bash sft.sh` runs supervised fine-tuning from a pretrained checkpoint.
- `bash slurm.sh` starts multi-node Slurm training.
- `python gradio/app.py /path/to/checkpoint` launches the demo UI.
- `cd eval/lmms-eval && pip install -e .` installs the evaluation harness.
- `bash eval/understanding_eval.sh` runs lmms-eval understanding benchmarks (edit model path).
- `bash eval/geneval/generation.sh` generates Geneval images (edit `HF_HOME` and `MODEL`).

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/vars, `PascalCase` for classes, module names lowercase.
- Shell: keep scripts in `bash`, export env vars in uppercase (for example, `HF_HOME`, `IMG_FOLDER`).
- No repo-wide formatter; when editing `eval/lmms-eval`, follow its Black config (line length 240) and isort.

## Testing Guidelines
- There is no dedicated unit-test suite at the repo root.
- Use evaluation scripts in `eval/` for validation; include a quick inference smoke test when changing model loading or I/O.

## Commit & Pull Request Guidelines
- History favors short imperative messages (for example, "Add files via upload", "Update README.md"); keep subjects concise.
- PRs should include a summary, linked issues (if any), config changes, and evaluation or sample outputs for model behavior changes; add screenshots for Gradio UI updates.
- Do not commit large checkpoints or datasets; reference external storage instead.

## Environment & Data Paths
- Training and eval scripts expect `HF_HOME`, `IMG_FOLDER`, and `OUTPUT_FOLDER` to be set; update paths before running.
