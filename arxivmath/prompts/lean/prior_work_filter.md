# Task Description
I am constructing benchmark items for research-level mathematics.
In particular, we want to see whether LLMs can generate a Lean proof of a new mathematical result, given its formalization.

I have extracted an original theorem-style statement describing a major contribution from the abstract. You are checking whether the statement as is already appears in prior work, or whether it depends on the new contribution of the paper. To check this, you have access to the full paper text.

Discard the pair if the full paper gives any indication that prior work already proves (very similar variants of) the original statement, or that the original statement can be easily derived from prior work. For instance, if the paper states that the original statement is a straightforward application of a known result, then discard it. 

Discard the statement if there is any uncertainty. It is better to be strict and discard statements that are borderline than to keep statements where prior work clearly already gives the result or makes it easy to derive.

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

### Full paper text ###
{full_text}
