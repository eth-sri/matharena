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

## Overview

This README explains the Project Euler workflow in MathArena: adding new Project Euler problems, running models with code execution, submitting candidate answers to the Project Euler website, and updating the local gold answers afterward.

## Adding a New Problem

To add a newly released Project Euler problem, run:

```bash
uv run python scripts/euler/add_euler.py --problem_id 954
```

Here `--problem_id` is the actual Project Euler problem number, for example `954`.

This will:
- Add the problem statement to the `data/euler/euler/problems` directory.
- Update the number of problems in `configs/competitions/euler/euler.yaml` and insert today's date as a placeholder that you should verify.
- Update `data/euler/euler/source.csv`, which maps local MathArena problem ids to Project Euler ids in the form `eulerXXX`.
- Set the answer to "none" in the `answers.csv` file.
- Update `website/flaskr/static/data/competitions.json` and add the new problem with difficulty `TBD`.

The downloaded statements still contain HTML such as `<p>...</p>`. Clean those up manually before running evaluations.

## Running Models

Project Euler evaluations require code execution.

For most models, code execution is performed remotely through Modal. Follow the Modal quickstart if you want to use that path: <https://modal.com/docs/guide>.

There is also a local Docker fallback. To enable it, build the image first:

```bash
docker build -t matharena-docker docker/
```

After your code-execution backend is set up, run the smoke tests:

```bash
uv run pytest tests/test_code_execution.py
```

Then run models with the normal competition runner:

```bash
uv run python scripts/run.py --models gemini/gemini-pro-2.5 --comp euler/euler
```

## Submitting Candidate Answers

After generating model outputs, submit the candidate answers to Project Euler to determine the correct answer. Install Playwright once if needed:

```bash
uv run playwright install
```

Then submit answers for a local MathArena problem id:

```bash
uv run python scripts/euler/project_euler_submit.py --problem_id 12
```

Important: this `--problem_id` is the local MathArena id from `data/euler/euler/source.csv`, not the Project Euler problem number. For example, local id `12` currently maps to `euler954`.

The submission script expects:
- `EULER_USERNAME`
- `EULER_PASSWORD`

It pre-fills answers from the saved runs in `outputs/euler/euler/...`. You will likely be asked to solve a CAPTCHA on login and possibly during submissions. Enter the CAPTCHA in the terminal, not in the browser.

The script uses the `MIN_DELAY` and `MAX_DELAY` constants in `scripts/euler/project_euler_submit.py` to sleep between submissions.

It stops once all candidate answers have been submitted or a correct answer has been accepted.

Submission logs are stored in `logs/euler/<problem_id>/` so the script can avoid resubmitting the same answers on later runs.

## Updating the Gold Answers and Regrading

If all submitted answers are incorrect, there is nothing else to do.

If one of the submitted answers is correct, update `data/euler/euler/answers.csv` with the accepted answer and then regrade:

```bash
uv run python scripts/regrade.py --comps euler/euler
```

After regrading, rerun any website postprocessing you need.
