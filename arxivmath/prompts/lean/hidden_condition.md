# Task Description
I am constructing benchmark items for research-level mathematics.
In particular, we want to see whether LLMs can generate a Lean proof of a new mathematical result, given its formalization.
The original statement and its formalization were extracted from the abstract. Unfortunately, we have found that many papers skip essential conditions in the abstract that are only mentioned in the full paper. This makes the original question incomplete, which in turn makes the formalized statement incorrect.

Your task:
- Discard the question if it is fundamentally flawed due to missing conditions that are only mentioned in the full paper. This can be **any** trivial condition, including not stating that the result only holds for $n > 1$, or that a certain object is nonempty, etc. If you, in any way believe the question is incomplete or incorrect as it stands, discard it. It is better to be strict and discard questions that are borderline than to keep questions that are clearly incomplete or incorrect.
- Keep the question if it is already accurate and central.

## Output Format

Respond only with a JSON object:

```json
{{
  "action": string,
  "rationale": string,
}}
```

- "action": "discard" | "keep"
- "rationale": short justification grounded in the paper's discussion of prior work

### Original statement ###
{original_statement}

### Formalized statement ###
{formalized_statement}

### Full paper text ###
{full_text}
