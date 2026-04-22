I am creating a mathematical benchmark for LLMs called ArXivMath. For this purpose, I am extracting questions from recent arXiv papers along with their answers. In particular, I gave an LLM the title and abstract of each paper and asked it to generate a question and answer pair about the paper's main result.

## Problem
However, I have noticed that a lot of the questions can be answered just by looking at prior work cited in the paper, without needing to understand the new contributions of the paper itself. For instance,
- If the question asks about an upper bound, but the only new contribution of the paper is to show that this upper bound is tight (i.e., the bound was derived in prior work), then the answer can be inferred from prior work.
- If the question generalizes a known result but the final numeric/algebraic answer is the same as in prior work, then the answer can be inferred from prior work.
-If the final answer of a  question was already correctly predicted by a conjecture recorded in the mathematical literature, then the answer can be inferred from prior work. 
Of course, there are many other examples of this phenomenon.

This is problematic because I want the benchmark to test understanding of new research contributions, not just knowledge of prior work. Since I solely evaluate correctness of the final answer, I cannot tell whether the model just guessed the answer from prior work or actually derived the new result (it is not given the paper text at evaluation time). Your job is to filter out such questions that can be answered solely based on prior work cited or discussed in the paper. In particular, if simple or trivial reasoning from prior work suffices to guess the answer to the question, then the question should be discarded. It is only when the final answer depends on genuinely new results from the paper that cannot be inferred solely from prior work, that the question should be kept.

## Instructions
- Discard the question if the full paper indicates the answer can be guessed from prior work cited or discussed in the paper (e.g., the paper shows a known bound is tight, or it generalizes earlier results but yields the same final numeric/algebraic answer). For instance, if the same final answer was obtained in prior work only in a more limited setting, but the new paper extends it to a broader setting using new techniques, then discard the question.
- Keep the question if the full paper indicates the answer depends on genuinely new results that cannot be inferred/guessed from prior work. 
- Be strict in your filtering: I prefer to discard borderline cases rather than keep them.

## Output format
Return JSON with keys:
- "action": "discard" | "keep"
- "rationale": short justification grounded in the full paper's discussion of prior work

For instance,
{{
  "action": "discard",
  "rationale": "The paper states prior work already determined the bound, and this work only proves tightness, so the answer is implied by earlier results."
}}

Additional instructions:
1. Base your decision strictly on the content of the full paper (including its references and discussion of prior work).
2. Do not rely on external knowledge or assumptions beyond what is presented in the paper.
3. Do not edit or rewrite the question; only decide keep vs discard.

### Current question ###
{question}

### Current answer ###
{answer}

### Full paper text ###
{full_text}
