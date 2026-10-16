<div align="center">
    <h1><img height="150px" src="./images/matharena_icon.png" alt="MathArena"><br>MathArena</h1>

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

MathArena (NeurIPS D&B '25) is a platform for evaluation of LLMs on latest math competitions and olympiads. It is hosted on [matharena.ai](https://matharena.ai/). This repository contains all code used for model evaluation. This README explains how to run your models. For more details about other aspects of the project, such as adding new competitions, please refer to the specific README files in the `readmes/` folder. You can find logs from our evaluation containing full reasoning traces (if available) and solutions produced by the models on our HuggingFace page: [https://huggingface.co/MathArena](https://huggingface.co/MathArena).

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

Before running harness models, build the Docker image and prepare its cache once from the repository root:

```bash
bash scripts/build_harness_arxivlean_image.sh
```

Execute the following command to evaluate a model on a competition:
```bash
uv run python scripts/run.py --comp path/to/competition --models path/to/model1
```
- `path/to/competition`: Relative path from the `configs/competitions` folder to the competition config file (excluding the `.yaml` extension).
- `path/to/model1`: Relative path (or multiple) from the `configs/models` folder to the model config file (excluding the `.yaml` extension). See [Adding a Model/Agent](#adding-a-model) below for model config file structure.

The scripts `run_all_fa.sh` and `run_all_kangaroo.sh` provide convenient shortcuts to run a model on all current non-deprecated competitions. They can be executed as follows:
```bash
bash scripts/run_all_fa.sh path/to/model
```

For ArXivLean September, follow the [Lean 4.34.1 setup instructions](readmes/README_arxivlean_september_evaluation.md) before running the evaluation. Earlier editions use the [legacy setup instructions](readmes/README_arxivlean.md).

**Example:**
```bash
uv run python scripts/run.py --comp aime/aime_2025 --models openai/gpt-4o 
```

**Additional Flags:**
- `--n`: Number of runs per problem (default: 4).
- `--redo-all`: Ignore existing runs for this model and rerun everything (default: false, continues from existing runs found in `outputs/`).
- `--problems`: One-based indices of problems to run (default: runs all problems).
- `--container-runtime docker|podman`: Container backend for harness runs and judging (default: Docker). Docker and Podman use separate image and cache stores.

For rootless Podman, build the runtime in Podman’s storage and select that backend when running:

```bash
bash scripts/build_harness_arxivlean_image.sh --container-runtime podman
uv run python scripts/run.py --comp arxiv/september --models openai/gpt-61-sol --container-runtime podman --n 1
```

September’s ArXivMath and BrokenArXiv harness configurations allow 12 hours and $100 per attempt, followed by a five-minute final-answer grace period at the dollar limit. ArXivLean September sets a $100 per-attempt limit and no overall time limit. Costs are estimates from token usage and the model config’s prices; an in-flight request and the configured grace period can exceed the nominal dollar limit.

### Current Website Competitions
The table below maps all non-deprecated competitions currently shown on the website to their competition config. `Requires judging` indicates whether the config requires a separate `scripts/judge/judge.py` pass.

| Website section | Website competition | Competition config | Requires judging |
| --- | --- | --- | --- |
| ArXivMath | 08/2026 | `arxiv/august.yaml` | Yes |
| ArXivMath | 09/2026 | `arxiv/september.yaml` | Yes |
| BrokenArxiv | 08/2026 | `arxiv_false/august.yaml` | Yes |
| BrokenArxiv | 09/2026 | `arxiv_false/september.yaml` | Yes |
| ArXivLean | 09/2026 | `arxivlean/september.yaml` | No (Lean comparator) |
| Visual Math | Kangaroo 2025 1-2 | `kangaroo/kangaroo_2025_1-2.yaml` | No |
| Visual Math | Kangaroo 2025 3-4 | `kangaroo/kangaroo_2025_3-4.yaml` | No |
| Visual Math | Kangaroo 2025 5-6 | `kangaroo/kangaroo_2025_5-6.yaml` | No |
| Visual Math | Kangaroo 2025 7-8 | `kangaroo/kangaroo_2025_7-8.yaml` | No |
| Visual Math | Kangaroo 2025 9-10 | `kangaroo/kangaroo_2025_9-10.yaml` | No |
| Visual Math | Kangaroo 2025 11-12 | `kangaroo/kangaroo_2025_11-12.yaml` | No |
| Final-Answer Comps | AIME 2026 | `aime/aime_2026.yaml` | No |
| Final-Answer Comps | HMMT Feb 2026 | `hmmt/hmmt_feb_2026.yaml` | No |
| Final-Answer Comps | Apex | `apex/apex_2025.yaml` | No |
| Final-Answer Comps | Apex Shortlist | `apex/shortlist_2025.yaml` | No |
| Proof-Based Comps | USAMO 2025 | `usamo/usamo_2025.yaml` | Yes |
| Proof-Based Comps | IMO 2025 | `imo/imo_2025.yaml` | Yes |
| Proof-Based Comps | IMC 2025 | `imc/imc_2025.yaml` | Yes |
| Proof-Based Comps | Miklos Schweitzer 2025 | `miklos/2025.yaml` | Yes |
| Proof-Based Comps | Putnam 2025 | `putnam/putnam_2025.yaml` | Yes |
| Proof-Based Comps | USAMO 2026 | `usamo/usamo_2026.yaml` | Yes |
| Project Euler | Project Euler | `euler/euler.yaml` | No |

The September editions each contain 53 problems and load directly from Hugging Face:
[ArXivMath](https://huggingface.co/datasets/MathArena/arxivmath-0926),
[BrokenArXiv](https://huggingface.co/datasets/MathArena/brokenarxiv-0926), and
[ArXivLean](https://huggingface.co/datasets/MathArena/arxivlean-0926).
Saved model outputs are published in the corresponding
[ArXivMath](https://huggingface.co/datasets/MathArena/arxivmath-0926_outputs),
[BrokenArXiv](https://huggingface.co/datasets/MathArena/brokenarxiv-0926_outputs), and
[ArXivLean](https://huggingface.co/datasets/MathArena/arxivlean-0926_outputs) output datasets.
Older competition configs remain available for reproducing earlier evaluations.

Note: Since we do not publish the correct answers for Project Euler problems, they cannot be directly judged as correct or incorrect without adapting `data/euler/euler/answers.csv` with the correct answers.
Putnam, IMC, IMO 2025, USAMO 2025, Miklos Schweitzer 2025 were graded using human verification. They can therefore not be graded automatically. For these competitions, if you want to run them, you will have to adjust their config similar to the USAMO 2026 competition.

### Competitions Requiring Grading
For competitions requiring grading (including BrokenArXiv and USAMO), run:
```bash
uv run python scripts/judge/judge.py --comp path/to/competition
```
There are various agents available for judging, you can see example configs for a couple agents in `configs/judges/`. If you want to add a new agent, you can follow the examples.


### Seeing Results

Launch a local web server that inspects all successful runs that were saved to `output`: `uv run python app/app.py --comp path/to/competition`, and access it at [http://localhost:5001/](http://localhost:5001/). This shows the final answers but also full interactions with the model or all steps that an agent took (see for example the runs of `GPT-5 Agent` on `apex/apex_2025`). Warning signs for runs indicate potential problems and should be manually verified. Any warning is caused by one of the following problems:

  * 💀: parser threw an error or encountered something unexpected.
  * ⚠️: The correct answer might be present in the model answer, but it was not extracted.
  * ❕: Model likely hit max token limit.

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
    - **bigmodel**: `BIGMODEL_API_KEY`
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
  - `batch_processing`: If set to true, the model will be queried using batch processing. Available for OpenAI, Anthropic, and OpenRouter models.
  - `use_openai_responses_api`: If set to true, will use the OpenAI responses API (instead of chat completions).
  - `harness`: Run the model through `codex`, `claude`, `kimi`, `gravity`, `qwen`, `opencode`, `deepcode`, `grok`, `muse`, or `vibe`; requires the competition to opt in with `allow_harness: true`. Set it to `false` to opt out of a competition default.
  - `harness_version`: Optional exact CLI version; defaults to the latest release resolved at run startup.
  - `harness_config`: Docker image, resource limits, workspace, and harness authentication settings. Harness runs use minimal reproducible CLI context by default; set `minimal_context: false` only to restore a CLI's native skills and instructions.
  - Other model/provider specific parameters (`config`, `provider`, `reasoning`, etc.).

New model profiles include `openai/gpt-61-sol`, `anthropic/opus_55_high`,
`qwen/qwen3.8_max`, `deepseek/deepseek_v41_flash`, `xai/grok-47`,
`mistral/mistral_large_4`, and `gemini/gemini-4-argon`. The Argon profile uses
Google API authentication through Antigravity CLI; set `GOOGLE_API_KEY`.
Each profile specifies its provider, harness version, reasoning settings, and token prices.

See [Harness-backed model runs](readmes/README_harness.md) for configuration, API billing, isolation, and arXivLean tool integration details.

### Agents

Agents are defined via top-level config files (see e.g., `config/models/openai/gpt-5-agent.yaml`) that point to a pure model config, indicating the underlying LLM API used by the agent, and an agent scaffolding config which parametrizes the agents' workflow. 

To add a new scaffolding, follow the example of `solvers/selfcheck_agent.py` which uses utility functions from `base_agent.py`.


---
## 📚 Citation

```
@article{dekoninck2026matharena,
      title={Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs}, 
      author={Jasper Dekoninck and Nikola Jovanović and Tim Gehrunger and Kári Rögnvaldsson and Ivo Petrov and Chenhao Sun and Martin Vechev},
      year={2026},
      eprint={2605.00674},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.00674}, 
}
```

For BrokenArXiv dataset generation (`arxiv_false`), see the [source-based Codex pipeline and overhaul notes](readmes/README_brokenarxiv.md).
