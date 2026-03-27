# Verification Task

You are verifying a proposed question-answer pair.

Main question: Is this question answerable or are there missing elements? 
In other words, can the question be understood and answered without additional context or definitions?

Additionally, remove the question if any of the following criteria are met:
- The answer is $0$ or $1$, or the answer is the same as the variable in the question, e.g. "Find X in function of $n$" with answer "$n$" (small variations like $n+1$ are fine). This is too guessable and I want to focus on more complex questions.

---

Answer "keep": true only if the question is self-contained and answerable without missing definitions or context. Otherwise "keep": false.

## Output Format

Respond **only** with a JSON object:

```json
{{
  "keep": boolean
}}
```

If any criterion fails, output `"keep": false`.
If all criteria pass, output `"keep": true`.

---

# Proposed Question
{question}

# Proposed Answer
{answer}
