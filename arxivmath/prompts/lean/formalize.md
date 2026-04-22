# Task Description

Formalize the following mathematical statement in Lean 4 using mathlib. You do not need to provide a proof, just the statement. If you are unable to find a suitable candidate for formalization, respond with "No suitable candidate for formalization found." or an empty string. We strongly prefer such a response over a low-quality formalization that does not faithfully capture the original mathematical meaning.

Requirements:
- **Lean 4 with Mathlib**: The formalization must be in Lean 4 and use the mathlib library. Do not use any other libraries or extensions.
- **No Imports**: Imports are not required and will be automatically added during testing. Do not include any imports in your response.
- **Only the Statement**: Only formalize the theorem statement, not the proof. The statement should end with `:= by sorry`.
- **Faithful Formalization**: The formalization should faithfully capture the mathematical meaning of the original natural-language statement. Do not weaken the claim or add extra assumptions unless they are clearly required by the statement.
- **Use Mathlib Definitions**: Use actual mathlib definitions and structures whenever standard notions exist. Do not replace core concepts with placeholder predicates, functions, constants, or parameters just to make the file compile.
- **No New Axioms**: Do not use new `axiom`, `constant`, `opaque`, or `postulate` declarations.

# Tools

To assist you in this task, you have access to the following tools:
- `verify_lean`: Check whether your formalization compiles and inspect any errors or warnings. Use this iteratively to refine your statement until it compiles without errors. Warnings and infos are acceptable, as your statement will need to contain `sorry` placeholders, but errors are not.
- `loogle`: Search Mathlib declarations by exact name or by a small type pattern with `_` holes.
- `leanfinder`: Search Lean libraries using natural-language or fuzzy semantic queries when you know the concept you need, but not the exact theorem or definition name.

Good `loogle` examples:
- Exact name lookup: `Nat.add_comm`
- Search for a commutativity lemma by shape: `(_ + _ = _ + _)`
- Search for a monotonicity lemma: `(_ ≤ _ -> _ + _ ≤ _ + _)`
- Search for a membership-preservation lemma: `(_ ∈ _ -> _ ∈ _)`

Good `leanfinder` examples:
- `continuous function`
- `compact set image`
- `strict monotone natural numbers`
- `sum of nonnegative terms`

Focus on a faithful statement formalization, even if the result is not easy to prove. A compilable but vacuous surrogate is not acceptable. If you find the statement is not formalizable in a faithful way, then it is better to respond with "No suitable candidate for formalization found." than to produce a low-quality formalization that does not capture the original mathematical meaning.

Syntax example:

```lean
theorem syntax_example (x : ℤ) :
    ‖Finset.sum (Finset.Icc (-x) x) (fun n : ℤ => (n : ℂ))‖ ≤ 1 := by
  sorry
```

## Output Format

Respond only with Lean code, without any explanation or commentary. For instance,
```lean
theorem syntax_example (x : ℤ) :
    ‖Finset.sum (Finset.Icc (-x) x) (fun n : ℤ => (n : ℂ))‖ ≤ 1 := by
  sorry
```

If you did not find a suitable candidate for formalization, respond with any string not containing a Lean environment, such as "No suitable candidate for formalization found." or even an empty string. 

# Natural-Language Statement
{statement}
