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

Execute the following command to evaluate a model on a competition:
```bash
uv run python scripts/run.py --comp path/to/competition --models path/to/model1
```
- `path/to/competition`: Relative path from the `configs/competition` folder to the competition config file (excluding the `.yaml` extension).
- `path/to/model1`: Relative path (or multiple) from the `configs/models` folder to the model config file (excluding the `.yaml` extension). See [Adding a Model/Agent](#adding-a-model) below for model config file structure.

The scripts `run_all_fa.sh` and `run_all_kangaroo.sh` provide convenient shortcuts to run a model on all current non-deprecated competitions. They can be executed as follows:
```bash
bash scripts/run_all_fa.sh path/to/model
```

**Example:**
```bash
uv run python scripts/run.py --comp aime/aime_2025 --models openai/gpt-4o 
```

**Additional Flags:**
- `--n`: Number of runs per problem (default: 4).
- `--redo-all`: Ignore existing runs for this model and rerun everything (default: false, continues from existing runs found in `outputs/`).
- `--problems`: One-based indices of problems to run (default: runs all problems).

### Current Website Competitions
The table below maps all non-deprecated competitions currently shown on the website to their competition config. `Requires judging` indicates whether the config requires a separate `scripts/judge/judge.py` pass.

| Website section | Website competition | Competition config | Requires judging |
| --- | --- | --- | --- |
| BrokenArxiv | 02/2026 | `arxiv_false/february.yaml` | Yes |
| BrokenArxiv | 03/2026 | `arxiv_false/march.yaml` | Yes |
| ArXivMath | 01/2026 | `arxiv/january.yaml` | No |
| ArXivMath | 02/2026 | `arxiv/february.yaml` | No |
| ArXivMath | 03/2026 | `arxiv/march.yaml` | No |
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

Agents are defined via top-level config files (see e.g., `config/models/openai/gpt-5-agent.yaml`) that point to a pure model config, indicating the underlying LLM API used by the agent, and an agent scaffolding config which parametrizes the agents' workflow. 

To add a new scaffolding, follow the example of `solvers/selfcheck_agent.py` which uses utility functions from `base_agent.py`.


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
