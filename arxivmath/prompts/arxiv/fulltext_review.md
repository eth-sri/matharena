You are reviewing a math question that was created from a paper abstract only.
This math question is supposed to be an extremely challenging problem that requires deep understanding of the paper's content. It is used to benchmark advanced AI systems.
You now have OCR of the full paper.

Your task:
- Discard the question if the full paper shows the question is not a major contribution of the paper, is incorrect, or is missing significant context (in particular, assumptions only mentioned in the full text and not in the abstract).
- Edit the question if it can be fixed by adding assumptions, clarifying scope, or specifying conditions that appear only in the full paper.
- Keep the question if it is already accurate and central.

Return JSON with keys:
- "action": "discard" | "edit" | "keep"
- "question": required only if action is "edit" (the fully edited question)
- "rationale": short justification grounded in the full paper

For instance,
{{
  "action": "edit",
  "question": "Edited question text here with necessary assumptions.",
  "rationale": "The original question lacked the assumption that X holds, which is clarified in the full paper."
}}

Additional instructions:
1. **Only make very small and necessary changes when editing.**
    The goal is to preserve as much of the original question as possible while ensuring correctness and completeness.
2. **Do not, under any circumstances, make the question easier.**
    Do not include any information that would simplify the question in any way. Only include strictly necessary context or assumptions. This is crucial. 
3. **The only reason to edit is to ensure all necessary assumptions from the full paper are included.**
    The question as stated might be ambiguous or incomplete without these assumptions. If the question is already complete and correct, keep it as is.
    Do not edit for style, clarity, or because you think it could be better phrased.
4. **Base your decisions strictly on the content of the full paper.**
    Do not rely on external knowledge or assumptions beyond what is presented in the paper.
5. **Do not reference the paper, authors, or phrases like "in this work" in your edits.**
    All necessary context must be included directly in the question.
6. **Machine-verifiable**  
   The answer must be suitable for **rule-based verification**, meaning it can be extracted and compared as a string or parsed LaTeX expression. NEVER ask the model to prove or explain anything.
7. **Answer remains identical**  
   When editing, ensure the answer does not change. The answer must remain exactly as it was originally provided. No variable names or symbols in the answer should be altered. Sometimes, the question will ask to post-process the answer into a specific format (e.g., compute the sum of the elements in this set). This is solely to make verification easier, and you must not give the model any additional information that would simplify the question.
8. **No simplifications**
    You only need to add assumptions in-so-far as they are strictly necessary for completeness.
    Do not add any hints, simplifications, or things that could be considered intermediate steps.
    The question is not supposed to match a single theorem/lemma number from the paper, but rather be a challenging problem that requires deep understanding of the entire paper. Therefore, do not restrict the question to a specific section or result unless absolutely necessary. The question needs to remain as challenging as possible, to fully benchmark advanced AI systems with deep understanding and reasoning capabilities.

### Current question ###
{question}

### Current answer ###
{answer}

### Full paper text ###
{full_text}
