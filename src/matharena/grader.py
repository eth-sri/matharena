"""
This module contains functions for grading model outputs.
"""

import re
from collections import defaultdict

from loguru import logger

from matharena.tools.submit_answer import check_hash_match

from matharena.utils import is_conversation_broken
from matharena.parser import WarningType, check_answers, extract_answer, extract_boxed_answer, parse_answer


def extract_numbers(text):
    """
    Extract numbers from the text.

    Args:
        text (str): The text to extract numbers from.

    Returns:
        list: A list of tuples, where each tuple contains the number string,
              start index, and end index.
    """
    # This regex handles integers and decimals.
    pattern = r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)"
    return [(m.group(), m.start(), m.end()) for m in re.finditer(pattern, text)]


def check_number_proximity_any_order(gold, model, threshold=20):
    """
    Check if the numbers from the gold answer appear close together
    in the model answer, regardless of their order.

    The function finds the smallest window in the model answer that contains
    at least one occurrence of each number from the gold answer. If the span
    of that window is less than or equal to the threshold, we flag it.

    Args:
        gold (str): The gold answer.
        model (str): The model's answer.
        threshold (int, optional): The proximity threshold. Defaults to 20.

    Returns:
        bool: True if the numbers are close together, False otherwise.
    """
    if len(gold) < 5:  # too easy, let's not do this
        return False
    gold_numbers = extract_numbers(str(gold))
    if not gold_numbers:
        return False  # Nothing to check if no numbers in the gold answer.

    threshold = max(2 * len(str(gold)), threshold)

    # We assume each number is considered only once.
    gold_set = set(num for num, _, _ in gold_numbers)

    model_numbers = extract_numbers(model)
    if not model_numbers:
        return False  # No numbers in model answer.

    # Gather occurrences of gold numbers in the model answer: (position, number)
    occurrences = [(start, num) for num, start, _ in model_numbers if num in gold_set]
    if not occurrences:
        return False  # None of the gold numbers appear in the model answer.

    # Sort occurrences by their position in the text.
    occurrences.sort(key=lambda x: x[0])

    # Use a sliding window to find the minimal span that covers all gold numbers.
    count = defaultdict(int)
    have = 0
    need = len(gold_set)
    min_window = float("inf")
    left = 0

    for right in range(len(occurrences)):
        pos_right, num_right = occurrences[right]
        count[num_right] += 1
        if count[num_right] == 1:  # first time this gold number appears in the window
            have += 1

        # Once the window covers all gold numbers, try to shrink it from the left.
        while have == need and left <= right:
            pos_left, num_left = occurrences[left]
            window_span = pos_right - pos_left
            if window_span < min_window:
                min_window = window_span
            # Move the left pointer; if removing this occurrence loses a needed number, update 'have'
            count[num_left] -= 1
            if count[num_left] == 0:
                have -= 1
            left += 1

    return min_window <= threshold


def check_all_numbers(text, gold_answer):
    """
    Check if any number in the text matches the gold answer.

    Args:
        text (str): The text to check.
        gold_answer (str): The gold answer.

    Returns:
        bool: True if a matching number is found, False otherwise.
    """
    if extract_answer(text, strict_parsing=True)[0] is not None:
        return False
    numbers = re.findall(r"\d+", text)
    return any(num == gold_answer for num in numbers)


def check_output_length(length):
    """
    Check if the output length is a power of 2 or 10 times a power of 2.

    Args:
        length (int): The length of the output.

    Returns:
        bool: True if the length is suspicious, False otherwise.
    """
    if length < 1000:
        return False
    if length % 1000 == 0:
        return True
    while length % 10 == 0 and length > 1:
        length /= 10
    while length % 2 == 0 and length > 1:
        length /= 2
    return length == 1


def extract_and_grade(messages, output_tokens, gold_answer, competition_config, debug_info=""):
    """
    Grade the model's messages against the gold answer.

    Args:
        messages (list): The list of message dictionaries from the model (clean format).
        gold_answer (str): The gold answer as string.

    Returns:
        tuple: (answer, is_correct, warning)
    """

    is_final_answer = competition_config.get("final_answer", True)
    use_strict_parsing = competition_config.get("strict_parsing", False)
    use_exact_match = competition_config.get("exact_match_parsing", False)

    gold_answer_is_list = is_final_answer and "," in gold_answer

    is_broken, reason = is_conversation_broken(messages)
    if is_broken:
        raise ValueError(f"Message list is broken: {reason}")

    last_message = messages[-1]["content"]
    if use_exact_match:
        model_answer, warning = extract_boxed_answer(last_message, list_answer=gold_answer_is_list)
    else:
        model_answer, warning = extract_answer(
            last_message, strict_parsing=use_strict_parsing, parse=True, list_answer=gold_answer_is_list
        )

    if gold_answer.startswith("hash:"):
        hval = gold_answer.split(":")[1]
        if check_hash_match(str(model_answer), hval):
            return model_answer, True, warning.value
        else:
            return model_answer, False, warning.value

    if use_exact_match:
        gold_is_int = re.fullmatch(r"\s*-?\d+\s*", str(gold_answer)) is not None

        def _normalize_exact_match(s):
            normalized = s.lower()
            if gold_is_int:
                normalized = re.sub(r"[a-z]", "", normalized)
                normalized = re.sub(r"[^0-9-]+", "", normalized)
            else:
                normalized = re.sub(r"[^a-z0-9]+", "", normalized)
            return normalized

        if model_answer is None:
            is_correct = False
        else:
            is_correct = _normalize_exact_match(str(model_answer)) == _normalize_exact_match(str(gold_answer))
    else:
        typed_gold_answer, _ = parse_answer(gold_answer, list_answer=gold_answer_is_list)
        is_correct = check_answers(model_answer, typed_gold_answer)

    if not is_correct and check_output_length(output_tokens):
        logger.warning(
            f"[{debug_info}] Model output length {output_tokens} is of the form 10**k * 2**n. This might indicate it hit the token limit."
        )
        warning = WarningType.MINOR  # model just didn't have time, any error could have been caused by this
    elif not is_correct and not use_exact_match and check_all_numbers(last_message, gold_answer):
        logger.warning(
            f"[{debug_info}] Model answer: {model_answer} is not equal to gold answer: {gold_answer} even though model output contains the gold answer."
        )
        warning = max(warning, WarningType.POSSIBLE)
    elif not is_correct and check_number_proximity_any_order(gold_answer, last_message):
        logger.warning(
            f"[{debug_info}] Numbers appearing in gold answer appear close together in model answer, but answer was incorrect."
        )
        warning = max(warning, WarningType.POSSIBLE)
    elif len(last_message) == 0:
        logger.warning(f"[{debug_info}] Empty message found.")
        warning = WarningType.MAJOR
    return model_answer, is_correct, warning.value
