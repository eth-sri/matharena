# Verification Task

You are verifying an original theorem statement and a perturbed theorem statement.
These statements were extracted from a research paper abstract, and the perturbed statement is designed to be false in light of the original statement.

Keep the pair only if all of the following are true:
- both statements are self-contained and understandable without the abstract. In particular, neither can refer to the abstract or each other. 
- the original statement is theorem-like and specific
- assuming the original statement is true, the perturbed statement is definitely false
- the perturbed statement is still plausible enough that one might imagine it being true if they didn't know the original statement

Discard if there is any meaningful ambiguity about the original statement, the perturbed statement, or the falsity of the perturbed statement given the original statement. Be strict.

## Output Format

Respond only with a JSON object:

```json
{{
  "keep": boolean
}}
```

# Original Statement
{original_statement}

# Perturbed Statement
{perturbed_statement}

# Claimed Falsity Explanation
{falsity_explanation}
