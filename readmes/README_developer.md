<div align="center">
    <h1><img height="150px" src="../images/matharena_icon.png" alt="MathArena"><br>MathArena</h1>

  <a href="https://www.python.org/">
<img alt="Build" src="https://img.shields.io/badge/Python-3.12-1f425f.svg?color=blue">
  </a>
  <a href="https://opensource.org/licenses/MIT">
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
  <a href="https://huggingface.co/MathArena">
<img alt="MathArena Datasets" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Matharena-ffc107?color=ffc107&logoColor=white">
  </a>
</div>


## 👋 Overview

MathArena (NeurIPS D&B '25) is a platform for evaluating LLMs on recent math competitions and olympiads. It is hosted on [matharena.ai](https://matharena.ai/). This repository contains the full evaluation codebase.

This README is the developer-oriented guide. It focuses on:
- running existing competitions locally,
- inspecting and postprocessing runs,
- adding new models or agents, and
- adding new competitions, including competitions that require judging.

For normal usage, start with the main [README](../README.md). You can find logs from our evaluations, including reasoning traces when available, on [https://huggingface.co/MathArena](https://huggingface.co/MathArena).

## 📑 Table of Contents
- [Installation](#-installation)
- [Running an Eval](#-running-an-eval)
  - [What This Does](#what-this-does)
  - [Regrading Existing Runs](#regrading-existing-runs)
  - [Inspecting and Debugging Runs](#inspecting-and-debugging-runs)
  - [Postprocessing for the Website](#postprocessing-for-the-website)
  - [Uploading Outputs to Hugging Face](#uploading-outputs-to-hugging-face)
- [Adding a New Model/Agent](#-adding-a-new-modelagent)
  - [Agents](#agents)
- [Adding a Competition](#-adding-a-competition)
  - [Dataset Format](#dataset-format)
  - [Competition Config](#competition-config)
  - [Local Dataset Layout](#local-dataset-layout)
  - [Uploading a Competition Dataset](#uploading-a-competition-dataset)
  - [Competitions Requiring Judging](#competitions-requiring-judging)
- [Citation](#-citation)

---
## 🚀 Installation

MathArena uses [UV](https://github.com/astral-sh/uv) to manage dependencies. If you want to run local models, uncomment the vllm installation in `pyproject.toml`.

### Install UV

- **macOS and Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows:**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### Alternative installation

As an alternative to UV, you can also create a conda environment and install the package as follows:
```bash
conda create -n matharena python=3.12
conda activate matharena
python -m pip install -e .
```
If you choose this option, disregard `uv run` in all instructions and use python directly instead.

---
## 🏃 Running an Eval

Execute the following command to evaluate a model on a competition:
```bash
uv run python scripts/run.py --comp path/to/competition --models path/to/model1
```
- `path/to/competition`: Relative path from the `configs/competitions` folder to the competition config file, without the `.yaml` extension.
- `path/to/model1`: Relative path from the `configs/models` folder to the model config file, without the `.yaml` extension. You can pass multiple model configs.

**Example:**
```bash
uv run python scripts/run.py --comp aime/aime_2026 --models openai/gpt-4o
```

**Additional Flags:**
- `--n`: Number of runs per problem (default: 4).
- `--redo-all`: Ignore existing runs for this model and rerun everything (default: false, continues from existing runs found in `outputs/`).
- `--problems`: One-based indices of problems to run (default: runs all problems).

### What This Does

This command instantiates the runner, loads the competition problems from `dataset_path`, creates the requested solver, and executes `n` attempts per problem. The solver is either a pure model or an agent scaffold built on top of a model config.

Each run is then parsed and graded, and the normalized results are written under `outputs/`.

There are several layers of retries during a run to handle rate limits and transient API failures. `run.py` may still finish without producing all requested attempts. Re-running the same command continues from the successful runs already present in `outputs/` unless you pass `--redo-all`.

### Regrading Existing Runs

Running `uv run python scripts/regrade.py` can be used to update saved runs in several ways:
- Update formatting inconsistencies in serialized runs, most importantly model interactions.
- Rerun parsing and grading on existing model interactions (useful if parser/grader have been patched after the run).
- Recompute costs based on token usage (useful if API costs have been updated after the run).

For example, to regrade all saved Project Euler runs with default settings:
```bash
uv run python scripts/regrade.py --comps euler/euler
```

Another useful helper is `scripts/nuke_single_run.py`, which removes a single saved attempt from a run file in `outputs/`.

### Inspecting and Debugging Runs

- `logs/status` shows the current progress of active runs.
- `logs/requests` stores verbatim API requests and is useful when debugging provider-specific issues.
- `logs/broken_runs` contains runs that could not be serialized correctly.
- `outputs/` contains the canonical normalized run artifacts.

To inspect completed runs in a browser, launch the local inspection app:
```bash
uv run python app/app.py --comp path/to/competition
```
and open [http://localhost:5001/](http://localhost:5001/).

This app shows final answers, full traces, grading output, and parser warnings. Warning icons indicate one of the following:

  * 💀: parser threw an error or encountered something unexpected.
  * ⚠️: The correct answer might be present in the model answer, but it was not extracted.
  * ❕: Model likely hit max token limit.

If you find issues, either delete the corresponding output file, remove a specific run with `scripts/nuke_single_run.py`, or patch the parser/grader and rerun `scripts/regrade.py`. The inspection app also supports manual overrides for parsed correctness and judged scores.

### Postprocessing for the Website

To convert local outputs into the JSON files used by the website, run:
```bash
uv run python scripts/website/postprocess.py --comp path/to/competition
```

To run the website locally:
```bash
cd website
uv run python flaskr/app.py --port 5005
```

Note: our website code is not included in the public repository, so you can ignore the website-related instructions if you do not have access to that code.

### Uploading Outputs to Hugging Face

You can upload the model answers to HuggingFace as follows:
```bash
uv run python scripts/curation/upload_outputs.py --org your_org --repo-name your_repo_name --comp path/to/competition
```
This uploads the model outputs to a private dataset repository named `your_org/your_repo_name`. `path/to/competition` is the path under `configs/competitions`, without the `.yaml` extension.

For Project Euler-specific operational steps, see [README_euler.md](./README_euler.md).

---
## 🤖 Adding a New Model/Agent

To add a new model add a config file in the `configs/models` folder. Each config must include:

- **Required:**
  - `model`: Model name. Reasoning effort of OpenAI models can be set by appending `--[low/medium/high]` to the model name, e.g., `o3-mini--high`.
  - `api`: API provider. The API key should be defined as an environment variable when using the specified API. The supported options with their corresponding API keys are:
    - **xai**: `XAI_API_KEY`
    - **openai**: `OPENAI_API_KEY`
    - **together**: `TOGETHER_API_KEY`
    - **google**: `GOOGLE_API_KEY`
    - **anthropic**: `ANTHROPIC_API_KEY`
    - **glm**: `GLM_API_KEY`
    - **deepseek**: `DEEPSEEK_API_KEY`
    - **openrouter**: `OPENROUTER_API_KEY`
    - **vllm**: (runs locally; no API key required)
  - `human_readable_id`: A unique, descriptive identifier.
- **Optional Parameters:**
  - API settings like `temperature`, `top_p`, and `top_k`.
  - `max_tokens`: Max number of tokens for the model.
  - `concurrent_requests`: Number of parallel requests to API (default: 30).
  - `timeout`: Request timeout in seconds (default: 2000).
  - `max_retries`: Retry attempts to API (default: 50).
  - `read_cost` & `write_cost`: Cost per million tokens in USD for input and output tokens (default: 1 each).
  - `cache_read_cost`: Cost per million cached input tokens in USD (default: same as `read_cost`).
  - `date`: Release date of the model in the format "yyyy-mm-dd".
  - `batch_processing`: If set to true, the model will be queried using batch processing. Only available for OpenAI and Anthropic models.
  - `use_openai_responses_api`: If set to true, will use the OpenAI responses API (instead of chat completions).
  - Other model/provider specific parameters (`config`, `provider`, `reasoning`, etc.).

### Agents

Agents are defined via top-level config files such as `configs/models/openai/gpt-5-agent.yaml`. These point to:
- a base model config that defines the underlying API model, and
- an agent scaffold config that defines the workflow.

To add a new scaffolding, follow the example of `solvers/selfcheck_agent.py` which uses utility functions from `base_agent.py`.

---
## ➕ Adding a Competition

### Dataset Format
MathArena supports the addition of any benchmark or competition uploaded to HuggingFace (or locally saved using the `datasets` library) that has the following columns:
- `problem_idx` (int): The id associated with the problem.
- `problem` (str): The problem statement.
- `answer` (str, Optional): The answer to the problem. Required for competitions with final answers.
- `points` (int, Optional): Maximum score for the problem. Required for competitions without final answers.
- `sample_solution` (str, Optional): Sample solution. Only used for non-final-answer competitions.
- `sample_grading` (str, Optional): Example grading output. Only used for non-final-answer competitions.
- `grading_scheme` (list, Optional): Per-problem grading scheme. Only used for non-final-answer competitions.

See [Competitions Requiring Judging](#competitions-requiring-judging) for the judged-competition flow.

### Competition Config

Add a competition config under `configs/competitions/<group>/<name>.yaml`. The core fields are:
- `instruction`: Instructions for the model. *Must* require the final answer be in `\boxed{}`.
- `strict_parsing`: `true` for strict format matching (e.g., only `\boxed{43}` is accepted) or `false` for lenient parsing.
- `n_problems`: Total number of problems.
- `date`: Date of the competition, in the format "YYYY-MM-DD".
- `dataset_path`: Dataset path, usually a Hugging Face dataset such as `MathArena/usamo_2026`, or a local path under `data/`.
- `final_answer` (optional): Set to `false` for competitions that require a separate judging pass. Defaults to `true`.
- `judge_configs` (optional): List of judge config paths under `configs/`, required if you want to run `scripts/judge/judge.py`.
- `grading_scheme_creator_config` (optional): Config used by `scripts/judge/create_grading_scheme.py` when grading schemes are generated from ground-truth proofs.

### Local Dataset Layout

If you are building the dataset locally first, create a directory under `data/` with the following structure:

1. `problems/`
   Store each problem as a separate LaTeX file named `1.tex`, `2.tex`, ..., `k.tex`.
2. `answers.csv` for final-answer competitions
   Use columns `id` and `answer`.
3. `grading_scheme.json` for judged competitions
   This should be a list of objects with:
   - `id`: problem id,
   - `points`: maximum points for the problem,
   - `scheme`: either a grading-scheme string or a list of rubric items.

If `scheme` is a list, each rubric item should contain:
- `points`: points for that part,
- `title`: short unique label for the part,
- `desc`: description of what earns the points.

If you want to generate grading schemes automatically, add `ground_truth_proofs` for each judged problem and set `grading_scheme_creator_config` in the competition YAML.

### Uploading a Competition Dataset

Once the dataset is ready, you can upload it to Hugging Face:
```bash
uv run python scripts/curation/upload_competition.py --org your_org --repo-name your_repo_name --comp path/to/competition
```
This uploads the dataset to a private repository named `your_org/your_repo_name`. `path/to/competition` is the path under `configs/competitions`, without the `.yaml` extension.

### Competitions Requiring Judging

For competitions without final answers, the current recommended flow is:

1. Set `final_answer: false` in the competition config and add `judge_configs`.
2. Create `grading_scheme.json` in the local dataset directory.
3. If you have ground-truth proofs and want to bootstrap the grading scheme automatically, add `grading_scheme_creator_config` to the competition config and run:
```bash
uv run python scripts/judge/create_grading_scheme.py --comp path/to/competition
```
4. Run the competition normally:
```bash
uv run python scripts/run.py --comp path/to/competition --models path/to/model1
```
5. Run the judges:
```bash
uv run python scripts/judge/judge.py --comp path/to/competition
```
6. Inspect the judged outputs in `app/app.py` and manually override scores if needed.

See `configs/competitions/usamo/usamo_2026.yaml` for the current config pattern. Some older competitions in the repository still use legacy `judge_config` fields, but new competition configs should use `judge_configs` because that is what `scripts/judge/judge.py` consumes.

---
## 📚 Citation

```
@article{balunovic2025matharena,
  title = {MathArena: Evaluating LLMs on Uncontaminated Math Competitions},
  author = {Mislav Balunović and Jasper Dekoninck and Ivo Petrov and Nikola Jovanović and Martin Vechev},
  journal = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmark},
  year={2025}
}
```
