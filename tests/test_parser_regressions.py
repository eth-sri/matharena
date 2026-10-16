import pytest

from matharena.grader import extract_and_grade
from matharena.parser import check_answers, extract_answer, parse_answer


def assert_equivalent(model_answer, gold_answer, *, list_answer=False):
    parsed_model, _ = parse_answer(model_answer, list_answer=list_answer)
    parsed_gold, _ = parse_answer(gold_answer, list_answer=list_answer)
    assert check_answers(parsed_model, parsed_gold)


def test_factorized_polynomial_with_braced_exponent_matches_expanded_form():
    assert_equivalent("(N+1)^{2}(4N+1)", "4N^3+9N^2+6N+1")


def test_unbraced_fraction_denominator_with_symbolic_numerator():
    assert_equivalent(r"\frac{p(p+1)}2", r"\frac{p^2+p}{2}")


def test_floor_binomial_shorthand_and_equivalent_rhs():
    assert_equivalent(
        r"\left\lfloor\frac n2\right\rfloor\binom q2="
        r"\frac{\left\lfloor\frac n2\right\rfloor\,q(q-1)}2",
        r"\binom{q}{2}\left\lfloor\frac{n}{2}\right\rfloor",
    )


def test_pm_expands_inside_list_answers():
    assert_equivalent(
        r"2,3,\frac{3 \pm i\sqrt{3}}{2}",
        r"2,3,\frac{3+i\sqrt{3}}{2},\frac{3-i\sqrt{3}}{2}",
        list_answer=True,
    )


def test_repeated_boxed_list_on_final_line_is_not_duplicated():
    answer, _ = extract_answer(
        r"Thus answer: \boxed{(13,14,18)}. assistantfinal\boxed{(13,14,18)}",
        strict_parsing=True,
        parse=True,
        list_answer=True,
    )
    assert answer.kind == "tuple"
    assert len(answer.answers) == 3


def test_ordered_tuple_answer_preserves_order_and_delimiter_type():
    tuple_answer, _ = parse_answer("(2,3)")
    reversed_tuple, _ = parse_answer("(3,2)")
    set_answer, _ = parse_answer(r"\{2,3\}")
    bare_list, _ = parse_answer("2,3", list_answer=True)
    reversed_bare_list, _ = parse_answer("3,2", list_answer=True)

    assert tuple_answer.kind == "tuple"
    assert check_answers(bare_list, tuple_answer)
    assert not check_answers(reversed_tuple, tuple_answer)
    assert not check_answers(reversed_bare_list, tuple_answer)
    assert not check_answers(set_answer, tuple_answer)


def test_interval_answer_preserves_endpoint_openness():
    closed, _ = parse_answer("[0,1]")
    half_open, _ = parse_answer("(0,1]")

    assert closed.kind == "interval"
    assert check_answers(parse_answer("[0,1]")[0], closed)
    assert not check_answers(half_open, closed)


def test_asymptotic_prefix_and_little_o_are_removed():
    assert_equivalent(
        r"N(X)=\frac{27}{\pi^{2}}X^{2}+o(X^{2})\quad (X\to\infty)",
        r"\frac{27}{\pi^2}X^2",
    )


def test_exact_expression_before_boxed_approximation_is_used():
    answer, _ = extract_answer(
        r"\boxed{\Pr(T>n)=\frac{\binom{42}{n}}{\binom{54}{n}}}"
        r" and finally \boxed{\mathbb E[T]=\frac{737}{39}\approx 18.9}",
        strict_parsing=True,
        parse=True,
    )
    assert check_answers(answer, parse_answer(r"\frac{737}{39}")[0])


def test_decimal_only_final_box_prefers_previous_exact_box():
    answer, _ = extract_answer(
        r"Thus final answer: \boxed{\frac{737}{39}}. "
        r"Equivalently, on average this is about \boxed{18.90}.",
        strict_parsing=True,
        parse=True,
    )
    assert check_answers(answer, parse_answer(r"\frac{737}{39}")[0])


def test_inequality_answer_is_not_collapsed_to_rhs():
    answer, is_correct, _ = extract_and_grade(
        [{"role": "assistant", "content": r"\[\boxed{b_1(\tilde M)\ge 2}\]"}],
        100,
        "4",
        {"final_answer": True, "strict_parsing": False},
    )
    assert str(answer) == "b_{1}(M*tilde) >= 2"
    assert not is_correct


def test_tagged_box_tokens_are_treated_as_boxed_answers():
    answer, _ = extract_answer(
        r"<|begin_of_box|>D<|end_of_box|>",
        strict_parsing=True,
        parse=False,
    )
    assert answer == "D"


@pytest.mark.parametrize("left", [0, 0.0])
@pytest.mark.parametrize("right", [0, 0.0])
def test_numeric_zero_equals_zero(left, right):
    assert check_answers(left, right)


@pytest.mark.parametrize("nonzero", [41, -1, 1e-12, -1e-12])
def test_numeric_zero_does_not_match_nonzero(nonzero):
    assert not check_answers(nonzero, 0)
    assert not check_answers(0, nonzero)


def test_boxed_zero_is_correct_for_zero_gold():
    answer, is_correct, warning = extract_and_grade(
        [{"role": "assistant", "content": r"\boxed{0}"}],
        100,
        "0",
        {"final_answer": True, "strict_parsing": False},
    )
    assert str(answer) == "0"
    assert is_correct
    assert warning == 0
