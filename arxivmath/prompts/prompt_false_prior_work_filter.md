# Task Description
I am constructing benchmark items for research-level mathematics.
In particular, we want to measure how often LLMs claim to provide proofs of false statements that are very plausible and mathematically natural, but are false in light of a major contribution from a recent research paper.

I have extracted an original theorem-style statement describing a major contribution from the abstract, and a perturbed statement that is very plausible but false in light of the original statement. You are checking whether the truth value of a perturbed false statement can be determined from prior work discussed in the paper.

Discard the pair if the full paper gives any indication that prior work already makes the perturbed statement clearly true or clearly false. 
Keep the pair only if the paper indicates that recognizing the perturbed statement as false depends on the new contribution rather than prior work.

Discard the pair if there is any uncertainty. It is better to be strict and discard pairs that are borderline than to keep pairs where prior work might be interpreted as making the perturbed statement clearly true or false.

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

### Perturbed statement ###
{perturbed_statement}

### Falsity explanation ###
{falsity_explanation}

### Full paper text ###
{full_text}
