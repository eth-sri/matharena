from matharena.grader import extract_and_grade
from matharena.utils import ensure_final_response_message, normalize_conversation


def test_ensure_final_response_message_appends_empty_response_after_trailing_cot():
    raw_messages = [
        {"role": "user", "content": "Put your final answer within \\boxed{}."},
        {"role": "assistant", "type": "cot", "content": ""},
    ]

    patched_messages = ensure_final_response_message(raw_messages)

    assert len(patched_messages) == 3
    assert patched_messages[-1] == {"role": "assistant", "content": ""}


def test_extract_and_grade_treats_trailing_cot_as_empty_incorrect_response_after_patch():
    raw_messages = [
        {"role": "user", "content": "Put your final answer within \\boxed{}."},
        {"role": "assistant", "type": "cot", "content": ""},
    ]

    clean_messages = normalize_conversation(ensure_final_response_message(raw_messages))
    answer, is_correct, _warning = extract_and_grade(
        clean_messages,
        256000,
        "42",
        {"final_answer": True, "strict_parsing": True},
    )

    assert clean_messages[-1] == {"role": "assistant", "type": "response", "content": ""}
    assert answer is None
    assert is_correct is False
