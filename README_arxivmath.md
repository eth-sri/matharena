This README file provides instructions specific to the arxivmath competition.
We first explain how to parse and add a new month of problems, then we briefly touch upon running models.

## Adding a new month of problems
First, remove the `arxivmath/paper` folder if you have it from before (or move it somewhere else).
When there is a new month of problems, you should run the following commands in order:
```bash
uv run python arxivmath/download_arxiv_math.py --from 2026-02-01 --until 2026-02-28
bash arxivmath/create.sh
```
For the last command, you need to set up DeepSeek-OCR serving. You can do this by running the following command in a separate terminal:
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
Note: the exact parameters for `vllm serve` may vary depending on your hardware. In particular, adjust tensor parallel size and GPU memory utilization as needed.

After these commands, you have performed all automated steps to prepare the new month of problems. Now, we enter the manual stage. For this reason, first open the app:
```bash
uv run python arxivmath/app.py --check-kept
```
This opens a localhost browser that allows you to review the problems, edit them, or even discard them. In the manual check, make sure to:
1. Remove questions that are guessable. For instance, problems where the answer is 0 or 1, or where the question has the type "Find X in function of $n$" and the answer is $n$.
2. Remove questions that are trivial. Sometimes, the model manages to mess up and remove all difficulty from a question. For instance, this could be a question of the type "Suppose X is bigger than 10 and smaller than 10. What is X?" with answer "X is 10". Note: the triviality of the problem is often a bit hidden.
3. Remove questions with non-unique answers. These questions often contain phrases like "For which $n$ is it proven/established that, ...". These depend on the paper's context and are not well-defined standalone.
4. Remove questions with unparseable answers. These are often some difficult set with various constraints, or some equivalence class of objects. Sometimes these are editable to make them more manageable as well.

I usually remove around 60-80% of questions in the manual pass.

Finally, export the questions:
```bash
uv run python arxivmath/export_accepted_questions.py --out-dir data/arxiv/february
```

Then, create a new config in `configs/competitions/arxiv` for the new month, based on the previous months.Running the models is then done as usual with the competition framework. If you use the tools that allow for search and paper reading, you should also run the DeepSeek-OCR server as described above.

While running the models, accurately check their answers and correct any further mistakes. I usually go in more detail over questions that models got all wrong, as this might be caused by noise. I check the questions they only got partially right for parsing or ambiguity issues. Finally, I check the questions they always got right for triviality. Remove the questions by using the `nuke_problems.py` script in the competition framework, e.g.:
```bash
uv run python scripts/curation/nuke_problems.py arxiv/february 5
```

The answers are quite frequently more difficult to extract than our other competitions. To mitigate parsing issues, run:
```bash
uv run python scripts/check.py --comp arxiv/february
```
This will run an LLM-as-a-judge to verify the answers. You can check in the app which problems had mismatches and correct them manually.