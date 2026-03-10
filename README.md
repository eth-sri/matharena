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

MathArena (NeurIPS D&B '25) is a platform for evaluation of LLMs on latest math competitions and olympiads. It is hosted on [matharena.ai](https://matharena.ai/). This repository contains all code used for model evaluation. This README explains how to run your models or add a new competition. You can find logs from our evaluation containing full reasoning traces (if available) and solutions produced by the models on our HuggingFace page: [https://huggingface.co/MathArena](https://huggingface.co/MathArena).

## 📑 Table of Contents
- [Installation](#-installation)
  - [Install UV](#install-uv)
  - [Alternative installation](#alternative-installation)
- [Running an Eval](#-running-an-eval)
  - [What Does This Do?](#what-does-this-do)
  - [Updating Runs](#updating-runs)
  - [Tracking Progress and Debugging Runs](#tracking-progress-and-debugging-runs)
  - [Adding Runs to the Website](#adding-runs-to-the-website)
  - [Uploading Answers to HuggingFace](#uploading-answers-to-huggingface)
  - [Project Euler](#project-euler)
- [Adding a New Model/Agent](#-adding-a-new-modelagent)
  - [Agents](#agents)
- [Adding a Competition](#-adding-a-competition)
  - [Competition Format](#competition-format)
  - [Configuration](#configuration)
  - [Manual Curation and Creation](#manual-curation-and-creation)
    - [Setting Up Competition Files](#setting-up-competition-files)
    - [Verifying Problem Statements](#verifying-problem-statements)
    - [Upload to HuggingFace](#upload-to-huggingface)
  - [Competitions Requiring Grading](#competitions-requiring-grading)
- [Scripts](#-scripts)
  - [Creating a Leaderboard](#creating-a-leaderboard)
  - [Curation](#curation)
  - [Extraction](#extraction)
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
- `path/to/competition`: Relative path from the `configs/competition` folder to the competition config file (excluding the `.yaml` extension).
- `path/to/model1`: Relative path (or multiple) from the `configs/models` folder to the model config file (excluding the `.yaml` extension). See [Adding a Model/Agent](#adding-a-model) below for model config file structure.

**Example:**
```bash
uv run python scripts/run.py --comp aime/aime_2025 --models openai/gpt-4o 
```

**Additional Flags:**
- `--n`: Number of runs per problem (default: 4).
- `--redo-all`: Ignore existing runs for this model and rerun everything (default: false, continues from existing runs found in `outputs/`).
- `--problems`: One-based indices of problems to run (default: runs all problems).

### What Does This Do?

This instantiates a Runner (`runner.py`) which loads competition problems (from HuggingFace or locally) and instantiates a <b>Solver</b> corresponding to either a pure model (`solvers/pure_model_solver.py`) or an agent (`solvers/agent_pool.py`). See [Adding a Model/Agent](#adding-a-model) for more details on agents.

The runner prompts the LLM API (`api_client.py`) to solve each problem `n` times. Each run is then parsed (`parser.py`) and graded against the gold solution (`grader.py`). Finally, all data from runs (`runs.py`) is normalized into a common API-independent format and saved under `outputs/`.

*Note*: There are several layers of retries during one run, accounting for rate limiting and other API errors. Runs are not accepted if the model fails to report an answer; to make this less common, we reprompt the model one last time if no answer was reported (`solver.last_chance`). Still, `run.py` might finish without producing `n` runs for each problem. In this case repeat the run, which will by default not repeat the successful runs found in `outputs/`.

### Updating Runs 

Running `uv run python scripts/regrade.py` can be used to update saved runs in several ways:

- Update formatting inconsistencies in serialized runs, most importantly model interactions.
- Rerun parsing and grading on existing model interactions (useful if parser/grader have been patched after the run).
- Recompute costs based on token usage (useful if API costs have been updated after the run).

For a default run that regrades all of euler/euler with default parameters (N=4, all updates) run `uv run python scripts/regrade.py --comps euler/euler`.

Another useful script is `scripts/nuke_single_run.py` which given a path to a runs file in `outputs/` removes a specific run at a given index.

### Tracking Progress and Debugging Runs

There are several ways to track progress and debug runs:

1. Track files under `logs/status` which show an updated overview of the progress of all current runs.
2. Inspect `logs/requests` which verbatim logs each request made to an API in `api_client.py`. As final outputs are postprocessed to a common format, this can be useful to identify API-specific errors. 
3. Inspect `logs/broken_runs` for runs which unexpectedly could not be saved. 
4. Launch a local web server that inspects all successful runs that were saved to `output`: `uv run python scripts/app.py --comp path/to/competition`, and access it at [http://localhost:5001/](http://localhost:5001/). This shows the final answers but also full interactions with the model or all steps that an agent took (see for example the runs of `GPT-5 Agent` on `apex/apex_2025`). Warning signs for runs indicate potential problems and should be manually verified. Any warning is caused by one of the following problems:

  * 💀: parser threw an error or encountered something unexpected.
  * ⚠️: The correct answer might be present in the model answer, but it was not extracted.
  * ❕: Model likely hit max token limit.

If issues are found, delete all runs for that problem by deleting the corresponding output file or use `runs.py:drop_runs` for selective removal. After that, call `run.py again` or only repeat the grading using `scripts/regrade.py` as described above. If the parser requires a manual overwrite, you can edit `src/matharena/parse_manual.py` and add a key-value pair mapping the model solution to a parsable solution.

### Adding Runs to the Website

To postprocess results to add them to our website, you should run
```bash
bash scripts/website/postprocess.sh path/to/competition
```

You can test this by running a local instance of the MathArena website via
```
cd website
uv run python3 flaskr/app.py
```

### Uploading Answers to HuggingFace
You can upload the model answers to HuggingFace as follows:
```bash
uv run python scripts/curation/upload_outputs.py --org your_org --repo-name your_repo_name --comp path/to/competition
```
This will upload all model answers to a private repository named `your_org/your_repo_name`. `path/to/competition` is the relative path from the `configs/competition` folder to the competition folder (excluding the `.yaml` extension).

### Project Euler

For Project Euler, several additional steps need to be taken. Please check README_euler.md for full details.

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
  - `enable_anthropic_prompt_caching`: Enable Anthropic prompt-caching markers (`cache_control`) for Anthropic Messages API calls (default: true).
  - `use_openai_responses_api`: If set to true, will use the OpenAI responses API (instead of chat completions).
  - Other model/provider specific parameters (`config`, `provider`, `reasoning`, etc.).

### Agents

Agents are defined via top-level config files (see e.g., `config/models/openai/gpt-5-agent.yaml`) that point to a pure model config, indicating the underlying LLM API used by the agent, and an agent scaffolding config which parametrizes the agents' workflow. 

To add a new scaffolding, follow the example of `solvers/selfcheck_agent.py` which uses utility functions from `base_agent.py`.

---
## ➕ Adding a Competition

### Competition Format
MathArena supports the addition of any benchmark or competition uploaded to HuggingFace (or locally saved using the `datasets` library) that has the following columns:
- `problem_idx` (int): The id associated with the problem.
- `problem`(str): The problem statement.
- `answer` (str, Optional): The answer to the problem. Required for competitions with final answers.
- `points` (int, Optional): The number of points associated with the problem. Only required for competitions without final answers.
- `sample_solution` (str, Optional): Sample solution to the problem. Only required for competitions without final answers and during autograding.
- `sample_grading` (str, Optional): Example of how the grading format should look like. Only required for competitions without final answers and during autograding.
- `grading_scheme` (list, Optional): The grading scheme for the problem. Only required for competitions without final answers.
We refer to [the instructions regarding graded competitions](#competitions-requiring-grading) for the specific format of the grading scheme.

### Configuration
To set up MathArena for evaluation on the competition, you should add a competition config file in the `configs/competitions` folder with the following parameters:
- `instruction`: Instructions for the model. *Must* require the final answer be in `\boxed{}`.
- `strict_parsing`: `true` for strict format matching (e.g., only `\boxed{43}` is accepted) or `false` for lenient parsing.
- `n_problems`: Total number of problems.
- `date`: Date of the competition, in the format "YYYY-MM-DD".
- `dataset_path`: Path to the dataset uploaded on HuggingFace or stored locally.
- `final_answer` (optional): If set to false, the competition is one that is manually graded with judges. Defaults to true if not set.

### Manual Curation and Creation
To create a pipeline that enables quick curation and easy generation of new competitions, we describe our full process for dataset creation. Note that you do not have to follow these steps if you have another way to generate your benchmark in the appropriate format.

#### Setting Up Competition Files
In the `data/` folder, create a new directory for your competition with the following structure:
1. **Problems:**  
   - Create a subfolder `problems/` and add each problem as a separate LaTeX file named `1.tex`, `2.tex`, ..., `{k}.tex`, where `k` is the number of problems in your competition. You can skip a problem if you want/need to.
2. **Answers:**  
   - If the competition is one based on final answers, add an `answers.csv` file with columns `id` and `answer`.
     - `id`: The problem filename (without the `.tex` extension).
     - `answer`: The integer answer.
   - If the competition is evaluated using human judges, add a `grading_scheme.json` file. This file should consist of a list of dictionaries, each of which contain the following fields:
     - `id`: The problem filename (without the `.tex` extension).
     - `points`: The maximum number of points for the question.
     - `scheme`: A list of dictionaries, each containing substeps for which points are awarded. Each dictionary contains the following keys:
        - `points`: Points associated with this step.
        - `title`: Title of the step. Should be unique across all dictionaries in this scheme.
        - `desc`: Description of the step.

#### Verifying Problem Statements
Ensure your LaTeX problems compile correctly:
```bash
uv run python scripts/curation/check_latex.py --comp path/to/competition
```
Then, build the `latex/main.tex` to generate a PDF and confirm all problems appear as expected.

#### Upload to HuggingFace
Finally, you can upload the competition to HuggingFace:
```bash
uv run python scripts/curation/upload_competition.py --org your_org --repo-name your_repo_name --comp path/to/competition
```
This will upload all answers in the appropriate format to a private repository named `your_org/your_repo_name`. `path/to/competition` is the relative path from the `configs/competition` folder to the competition folder (excluding the `.yaml` extension). Thus, you need to have created the configuration file before uploading to HuggingFace.

### Competitions Requiring Grading
For competitions requiring human grading, we use the Open Proof Corpus repository: https://github.com/insait-institute/open-proof-corpus. This repository contains instructions to run models on questions and competitions and contains a nice grading interface for judges. It also contains a script that converts that format to the MathArena format. The result of this script should simply be copy-pasted to `outputs/path/to/competition` for use and display in this repository.

---
## 📜 Scripts
`scripts` contains various utility files that serve purposes from curation and verification of model outputs to extracting the raw data into latex tables. We briefly describe the purpose of the files that have not been explained yet here.

### Creating a Leaderboard
If you prefer to see a leaderboard with rigorous confidence intervals, you can run
```bash
uv run python scripts/extraction/leaderboard.py --comps path/to/competition1 path/to/competition2
```
This script has several additional important parameters:
- `--keep-comps` (bool): In addition to the average across the listed competitions, whether to also keep the results for each competition separately in the leaderboard.
- `--compute-variance` (bool): Whether to compute the confidence intervals. Note: computing confidence intervals is expensive and can take several minutes.
- `--alpha` (Default: 0.05): Significance level associated with the confidence intervals.
- `--human-quantiles` (list[float]): Which quantiles of human performance to add to the leaderboard. Is only possible for AIME, SMT, and HMMT 2025.

### Curation

After making a change to the parser, and before rerunning models, you can test your updates using:
```bash
uv run python scripts/curation/test_parser_changes.py
```
This script will automatically extract every possible model output in the `outputs` folder and verify that the stored results match the new parser's output. The script will list all outputs that do not satisfy this, essentially serving as a rigorous test for the new parser.

To verify with a judge model whether the rule-based parser returns the correct answers, one can run 
```bash
uv run python scripts/curation/judge_parser.py --comp aime/aime_2025
```
This will verify and check all decisions by the rule-based parser using Gemini-Flash-2.5. Outputs where the model disagrees with the parser are logged in `logs/parser_judge`.

### Extraction
This folder contains scripts that extract the answers from the raw data and creates plots and a leaderboard.

To compare model performance between several competitions in a plot, one can run
```bash
uv run python scripts/extraction/comparison.py --old-comps aime/aime_2024 hmmt/hmmt_feb_2024 --new-comps aime/aime_2025 hmmt/hmmt_feb_2025
```

To compare the spearman correlation between different competitions, run
```bash
uv run python scripts/extraction/rank_correlation.py --comps aime/aime_2025  hmmt/hmmt_feb_2025
```

To create a plot of the Pareto-Frontier of model performance vs cost or model release date, run
```bash
uv run python scripts/extraction/timeline.py --comps aime/aime_2025  hmmt/hmmt_feb_2025
```
Here, specify the `--cost` parameter if you prefer the Pareto-Frontier for cost.

To create a table containing results per category (Combinatorics, Number Theory, ...), run
```bash
uv run python scripts/extraction/type_scoring.py --comps aime/aime_2025  hmmt/hmmt_feb_2025
```

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