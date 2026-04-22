# Verification Task

You are verifying whether a proposed theorem statement is suitable for a benchmark measuring the ability of LLMs to prove formal theorems in Lean 4. You are given a candidate natural language statement extracted from a paper together with its proof. 

Your task is to determine whether this statement is appropriate for inclusion in the benchmark, based on the criteria below.

Keep the statement only if all of the following hold:
- **Self-Contained**: it is self-contained and understandable. All notation is defined or standard, and there are no hidden assumptions or conventions that would be necessary to understand the statement.
- **Self-Contained Proof**: the proof is self-contained and understandable. It mostly does not reference the paper, but can reference other results from the same paper if those results are themselves strong candidates for formalization, or simple equations, definitions, or limited references. In particular, any reference to a seemingly major contribution of the paper is not accepted.
- **Notation and Conventions**: it does not rely on undefined paper-specific notation or hidden conventions
- **MathLib Definitions**: the statement does not fundamentally depend on a definition that is not standard in the Lean4 Mathlib library.
- **MathLib Theory**: the statement appears in an area of mathematics that is already well-developed in Lean4 Mathlib, so that the proof is likely to be formalizable in Lean4 with Mathlib.

Do not reject a statement merely because it is not a main theorem.

Be strict. If there is meaningful ambiguity, discard it.

## Output Format

Respond only with JSON:

```json
{{
  "keep": boolean,
  "rationale": "short justification"
}}
```

# Statement
{statement}

# Proof
{proof}
