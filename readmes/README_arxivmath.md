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

This README covers the ArXivMath and BrokenArXiv curation workflow: downloading a new month of papers, running the automated extraction pipeline, manually reviewing candidate questions, exporting the accepted questions, and then evaluating models on the resulting competitions.

## Prerequisites

Before running the extraction pipeline, remove or archive any old `arxivmath/paper` directory so that the new month starts from a clean workspace.

The automated pipeline uses DeepSeek-OCR. Start that server in a separate terminal before running `arxivmath/create.sh` or `arxivmath/create_false.sh`:

```bash
vllm serve deepseek-ai/DeepSeek-OCR-2 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 4 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --enforce-eager \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```

Adjust those `vllm serve` parameters to match your hardware.

## Adding a New Month

Download the source papers for the target month and run the extraction pipelines. For example, for February 2026:

```bash
uv run python arxivmath/download_arxiv_math.py --from 2026-02-01 --until 2026-02-28
bash arxivmath/create.sh
bash arxivmath/create_false.sh # For BrokenArXiv.
```

The helper scripts currently run the full extraction stack with the model configs hardcoded in `arxivmath/create.sh` and `arxivmath/create_false.sh`. If you want a different model, edit those scripts or run the underlying commands manually.

## Manual Review

After the automated pass, review the generated candidates in the annotation app:

```bash
uv run python arxivmath/app.py --check-kept
uv run python arxivmath/app.py --check-kept --false # For BrokenArXiv.
```

This opens a local review UI where you can edit, accept, or discard items. During the manual pass, remove:

1. guessable questions,
2. trivial questions,
3. questions with non-unique or context-dependent answers,
4. questions whose answers are too hard to parse robustly.

I usually remove around 50% of questions in the manual pass.

## Exporting the Accepted Questions

Once the review is done, export the accepted questions:

```bash
uv run python arxivmath/export_accepted_questions.py --out-dir data/arxiv/february
uv run python arxivmath/export_false_proofs.py --out-dir data/arxiv_false/february # For BrokenArXiv.
```

Then copy the previous month's competition config and update it for the new month:
- `configs/competitions/arxiv/<month>.yaml`
- `configs/competitions/arxiv_false/<month>.yaml`

At minimum, update `n_problems`, `date`, and `dataset_path`. Also add the new month to `website/flaskr/static/data/competitions.json` if it should appear on the website.

## Running Models on the New Month

Run models as usual with the normal competition runner:

```bash
uv run python scripts/run.py --comp arxiv/february --models openai/gpt-5
```

For BrokenArXiv, remember that a separate judging pass is required:

```bash
uv run python scripts/judge/judge.py --comp arxiv_false/february
```

If you evaluate agents that rely on OCR or paper-reading tools, keep the DeepSeek-OCR server running while they execute.

## Reviewing Model Outputs and Cleaning the Dataset

After running models, inspect the outputs carefully. I usually:
- inspect all problems that every model got wrong for noise or extraction errors,
- inspect partially solved problems for parsing or ambiguity issues,
- inspect universally solved problems for hidden triviality.

If a problem should be removed, use `nuke_problems.py`, for example:

```bash
uv run python scripts/curation/nuke_problems.py arxiv/february 5
```

ArXivMath answers are often harder to parse than final answers from olympiad-style contests. After a run, open the local inspection app to review parser mistakes and manually override results where needed:

```bash
uv run python app/app.py --comp arxiv/february
```

If you patch the parser or grader, rerun:

```bash
uv run python scripts/regrade.py --comps arxiv/february
```
