I am creating a mathematical benchmark for LLMs called ArXivMath. For this purpose, I am extracting questions from recent arXiv papers along with their answers. In particular, I gave an LLM the title and abstract of each paper and asked it to generate a question and answer pair about the paper's main result.

## Problem
However, many paper on ArXiv are now generated with the aid of LLMs. This gives an unfair advantage to LLMs that were used to generate the paper, since they can easily answer questions about the paper's main result. In contrast, LLMs that were not used to generate the paper would have to understand the paper's content and derive the main result in order to answer questions about it. This creates a bias in favor of LLMs that were involved in the paper generation process, which undermines the goal of testing LLMs' ability to understand and reason about new research contributions.

## Instructions
Discard any paper that mentions the use of LLMs or AI tools in the paper generation process, in any way. It does not matter how large the acknowledgment of AI is, or whether it is in the main text, acknowledgments, or references. If there is any indication that AI was used in any part of the paper, discard the paper. Take the use of AI liberally: look for any mention of AI tools, include Claude, Anthopric, OpenAI, ChatGPT, LLaMA, Gemini, etc. If there is any mention of these tools in the paper, discard the paper.

## Output format
Return JSON with keys:
- "action": "discard" | "keep"
- "rationale": short justification grounded in the full paper's discussion of prior work

For instance,
{{
  "action": "discard",
  "rationale": "The paper explicitly mentions the use of ChatGPT in the acknowledgments section, indicating that AI tools were involved in the paper generation process."
}}

### Current question ###
{question}

### Current answer ###
{answer}

### Full paper text ###
{full_text}
