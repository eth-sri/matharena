# Task Description

You are checking whether a Lean 4 theorem statement faithfully formalizes a natural-language mathematical statement.

Keep the formalization only if:
- the Lean code clearly states the same theorem as the natural-language statement
- no essential assumptions have been added or removed
- the theorem is neither weaker nor stronger in a materially different way
- the imported framework and encoding choices are mathematically faithful

Be extremely diligent. If there is any meaningful ambiguity, discard the formalization. We much prefer false negatives to false positives. If you are unsure, discard it.

## Output Format

Respond only with JSON:

```json
{{
  "keep": boolean,
  "rationale": "short explanation"
}}
```

# Natural-Language Statement
{natural_statement}

# Lean Code
{lean_code}
