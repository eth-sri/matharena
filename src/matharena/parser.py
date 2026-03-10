"""This module provides functions for parsing mathematical expressions from text."""

import re
from enum import Enum
from fractions import Fraction
from functools import total_ordering
from typing import Any, Optional
from sympy import simplify, Symbol, Mul

import regex
import sympy
from loguru import logger
from sympy import N
from sympy.parsing.latex import parse_latex

from matharena.parse_manual import complete_mapper, manual_mapper


@total_ordering
class WarningType(Enum):
    """An enumeration for warning levels."""

    NONE = 0
    MINOR = 1
    POSSIBLE = 2
    MAJOR = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return self.value < other


def latex2sympy_fixed(latex: str):
    # if _integer is present, replace it with _{integer} for any integer
    latex = re.sub(r"_([0-9]+)", r"_{\1}", latex)
    latex_parsed = parse_latex(latex)
    # replace constants like pi and e with their numerical value
    known_constants = {"pi": sympy.pi, "e": sympy.E, "I": 1j, "i": 1j}

    # Replace any symbol in expr that is in our known_constants dictionary.
    expr = latex_parsed.xreplace(
        {s: known_constants[s.name] for s in latex_parsed.free_symbols if s.name in known_constants}
    )
    return expr


def remove_inner_boxed(match: str):
    """Removes inner `\boxed` or `\fbox` commands from a string.

    Args:
        match (str): The string to process.

    Returns:
        str: The string with inner `\boxed` or `\fbox` commands removed.
    """
    pattern = r"(\\boxed|\\fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, match))
    if not matches:
        return match
    for m in matches:
        match = match.replace(m.group(0), m.group(2))
    return match


def find_last_boxed_content(text: str, list_answer: bool = False) -> Optional[str]:
    """Finds the content of the last `\boxed` or `\fbox` command in a string.

    Args:
        text (str): The string to search.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the content of the last `\boxed` or `\fbox` command
            and a warning level.
    """
    pattern = r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None, WarningType.NONE

    if len(matches) > 1 and list_answer:
        # find all boxed content on the same line (no \n in between) as the last boxed
        split_text = text.split("\n")
        for i in range(len(split_text) - 1, -1, -1):
            matches_line = list(regex.finditer(pattern, split_text[i]))
            if len(matches_line) > 0:
                returned_boxed = ",".join([match.group(2) for match in matches_line])
                return remove_inner_boxed(returned_boxed), WarningType.POSSIBLE

    last_match = remove_inner_boxed(matches[-1].group(2))
    return last_match, WarningType.NONE


def extract_boxed_answer(text: str, list_answer: bool = False) -> Optional[str]:
    """Extracts the content of the last `\boxed` or `\fbox` command in a string.

    Args:
        text (str): The string to search.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the content of the last `\boxed` or `\fbox` command
            and a warning level.
    """
    answer, warning = find_last_boxed_content(text, list_answer)
    if answer is not None:
        return answer, warning
    else:
        return None, warning


def replace_and_or(s: str) -> str:
    """Replaces 'and' or 'or' with commas in a string.

    1) If 'and/or' (or their \text{} forms) is NOT right next to a comma
       (ignoring spaces) -> replace it by a single ','.
    2) Otherwise (comma already on at least one side) -> delete it.

    Args:
        s (str): The string to process.

    Returns:
        str: The processed string.
    """
    TOKEN = re.compile(
        r"""
        (?:\\text\s*\{\s*)?      # optional '\text{' and any leading blanks
        (and|or)                 # the word itself
        (?:\s*\})?               # optional closing '}' with any blanks
        """,
        re.I | re.VERBOSE,
    )
    # We build a fresh output string piece-by-piece so that each check
    # uses the **current** comma layout, not the one from the original text.
    out, idx = [], 0
    for m in TOKEN.finditer(s):
        start, end = m.span()
        # copy text *before* the token
        out.append(s[idx:start])

        # look to the left of the token, skipping blanks
        j = start - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        comma_left = j >= 0 and s[j] == ","

        # look to the right of the token, skipping blanks
        k = end
        while k < len(s) and s[k].isspace():
            k += 1
        comma_right = k < len(s) and s[k] == ","

        # choose replacement
        out.append("" if (comma_left or comma_right) else ",")
        idx = end  # advance cursor

    out.append(s[idx:])  # tail of string
    return "".join(out)


def extract_boxed_answer_parse(text: str, parse: bool = True, list_answer: bool = False) -> Optional[int]:
    """Extracts and parses the content of the last `\boxed` or `\fbox` command.

    Args:
        text (str): The string to search.
        parse (bool, optional): Whether to parse the answer. Defaults to True.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the parsed answer and a warning level.
    """
    answer, warning = extract_boxed_answer(text, list_answer)
    if answer is not None:
        if answer.count("=") > 1:
            warning = max(warning, WarningType.MAJOR)  # this is a major warning, we should not have more than one "="
        try:
            return sympy.Integer(int(answer)), warning
        except:  # noqa: E722
            # logger.info(f"Could not parse answer {answer} as integer")
            if parse:
                parsed_answer, warning = parse_answer(answer, list_answer=list_answer)
                return parsed_answer, warning
            return answer, warning
    return None, WarningType.MAJOR


def extract_last_integer(text: str) -> Optional[int]:
    """Extracts the last integer from a string.

    Args:
        text (str): The string to search.

    Returns:
        tuple: A tuple containing the last integer and a warning level.
    """
    pattern = r"\b\d+\b"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None, WarningType.MAJOR
    try:
        return int(matches[-1].group()), WarningType.MAJOR
    except Exception as e:
        logger.warning(f"Error extracting last integer: {e}")
        return None, WarningType.MAJOR


def extract_answer(text: str, strict_parsing: bool = True, parse: bool = True, list_answer: bool = False):
    """Extracts and parses the final answer from a string.

    Args:
        text (str): The string to search.
        strict_parsing (bool, optional): Whether to use strict parsing. Defaults to True.
        parse (bool, optional): Whether to parse the answer. Defaults to True.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the parsed answer and a warning level.
    """
    if text is None or len(text) == 0:
        return None, WarningType.MAJOR
    warning_old = WarningType.NONE
    if text in complete_mapper:
        text = complete_mapper[text]
        warning_old = WarningType.MAJOR
    text, warning = replace_unicode(text)
    warning = max(warning, warning_old)
    answer, warning_new = extract_boxed_answer_parse(text, parse, list_answer)
    if isinstance(answer, AnswerList) and len(answer.answers) == 1:
        answer = answer.answers[0]
    warning = max(warning, warning_new)
    if answer is not None or strict_parsing:
        return answer, warning

    return extract_last_integer(text)

def parse_answer(s: str, primitive_type: type = None, list_answer: bool = False):
    """Parses a string into a mathematical expression.

    Args:
        s (str): The string to parse.
        primitive_type (type, optional): The primitive type to parse into. Defaults to None.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the parsed answer and a warning level.
    """
    warning = WarningType.NONE
    if s in manual_mapper:
        logger.warning(f"Applying manual parsing to {s}")
        s = manual_mapper[s]
        warning = WarningType.MAJOR
    s = remove_invalid_characters(s)
    s = remove_outer_brackets(normalize_string(s, list_answer))
    # s = insert_implicit_mul(s)
    output, warning_new = ParseList.parse("(" + s + ")", primitive_type=primitive_type)
    warning = max(warning, warning_new)
    if output is None:
        logger.warning(f"Could not parse {s}, returning None")
        return None, max(warning, WarningType.MAJOR)
    if len(output) == 1:
        output = output[0]

    if isinstance(output, list) or isinstance(output, tuple):
        output = AnswerList(output)
    return output, warning


def normalize_string(s, list_answer=False):
    """Normalizes a string for parsing.

    Args:
        s (str): The string to normalize.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        str: The normalized string.
    """
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\Bigl", "").replace(r"\Bigr", "")
    s = s.replace(r"\bigl", "").replace(r"\bigr", "")
    s = s.replace(r"\Big", "").replace(r"\big", "").replace(r"\Large", "").replace(r"\large", "")
    s = remove_aligns(s)
    s = s.replace("[", "(")
    s = s.replace("]", ")")
    s = s.replace("\\{", "(")  # sets will be converted to lists
    s = s.replace("\\}", ")")  # sets will be converted to lists
    s = s.replace("$", "")
    s = s.replace("\\ ", " ")
    # remove hline and vline
    s = s.replace(r"\hline", "")
    s = s.replace(r"\vline", "")
    s = s.replace(r"\quad", " ")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("·", " \\cdot ")
    s = s.replace("^\\circ", " ")
    s = s.replace("^{\\circ}", " ")
    s = s.replace("\\displaystyle", "")
    s = s.replace("\\(", "(")
    s = s.replace("\\)", ")")
    s = s.replace("{,}", "")  # o4-mini does this
    # remove \\begin{anything} and \\end{anything}
    if s.endswith("."):
        s = s[:-1]

    if list_answer and s is not None:
        s = replace_and_or(s)

    if not list_answer:
        # replace something of the type integer,integer with integerinteger
        s = re.sub(r"(?<=\d),(?=\d)", "", s)
        s = s.replace("{,}", "")
    if list_answer:
        s = s.replace(";", ",")
        s = s.replace("{,}", ",")
    # if we see \sqrt 123ea pi\frac -> \sqrt{123ea}pi\frac
    if "\\sqrt " in s:
        s = re.sub(r"\\sqrt\s*([^\s{}]*)", r"\\sqrt{\1}", s)
    # remove everything that appears within \text{...}
    s = re.sub(r"\\text\{.*?\}", "", s)
    # replace \mathrm{...} with ...
    s = re.sub(r"\\mathrm\{(.*?)\}", r" \1 ", s)

    s = s.replace("F_{30}", "832040")  # Fibonacci number present in one problem
    if "=" in s:
        s = s.split("=")[-1]
    if r"\in" in s and list_answer:
        s = s.split(r"\in")[-1]

    if "\\approx" in s:
        s = s.split("\\approx")[0]
        if s.endswith("("):  # in case it was put in brackets
            s = s[:-1]
    return strip(s)


def remove_outer_brackets(s):
    """Removes the outermost matching brackets from the string if they encompass the entire string.

    Parameters:
    s (str): The input string potentially wrapped with brackets.

    Returns:
    str: The string with the outermost brackets removed if they match and encompass the entire string.
    """
    while True:
        if not s:
            return s
        opening = s[0]
        closing = s[-1]

        if opening == "(" and closing == ")":
            count = 0
            matched = True
            for i, char in enumerate(s):
                if char == opening:
                    count += 1
                elif char == closing:
                    count -= 1
                if count == 0 and i != len(s) - 1:
                    matched = False
                    break

            if matched:
                s = s[1:-1]
                continue
        break

    return s


def remove_aligns(s: str) -> str:
    """Removes `\\begin{align}` and `\\end{align}` environments from a string.

    Args:
        s (str): The string to process.

    Returns:
        str: The processed string.
    """
    # This pattern captures:
    #   \begin{align followed by any non-} characters (like align*, alignat, etc.)
    #   then any content (non-greedily) up to
    #   \\end{align...} with the same "align" prefix
    pattern = r"\\begin{align[^}]*}(.*?)\\end{align[^}]*}"

    # Use a callback to remove '&' from the matched group before returning it
    return re.sub(pattern, lambda m: m.group(1).replace("&", "").replace("\\\\", ""), s, flags=re.DOTALL)


def replace_unicode(text: str) -> str:
    """Replaces unicode characters with their LaTeX equivalents.

    Args:
        text (str): The string to process.

    Returns:
        tuple: A tuple containing the processed string and a warning level.
    """
    text_old = text
    text = text.replace("\u23a7", r"\boxed{")
    text = text.replace("\u23ab", r"}")
    text = text.replace("\n\u2502", r"\boxed{")
    text = text.replace("\u2502", r"}")
    text = text.replace("\n\u2503", r"\boxed{")
    text = text.replace("\u2503", r"}")
    text = text.replace("\n\uf8f0", r"\boxed{")
    text = text.replace("\uf8fb", r"}")
    warning = WarningType.NONE if text == text_old else WarningType.POSSIBLE
    text = text.replace("\u221a", r"\sqrt")  # these ones are for sure fine, no warning necessary
    text = text.replace("\u00d7", r"\cdot")
    text = text.replace("\u202f", r" ")
    text = text.replace("\u2212", "-")
    text = text.replace("\u03c0", r"\pi")
    return text, warning


def remove_invalid_characters(text):
    """Removes invalid characters from a string.

    Args:
        text (str): The string to process.

    Returns:
        str: The processed string.
    """
    text = re.sub(r"\\;", "", text)
    text = re.sub(r"\\:", "", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\\!", "", text)
    return text


def strip(s: str):
    s = s.strip()
    # be careful with this, it can also remove the "\" in "\begin" if just done with strip
    while s.startswith(r"\n"):
        s = s[2:]
    while s.endswith(r"\n"):
        s = s[:-2]
    while s.startswith("\\ "):
        s = s[2:]
    # if s starts with any thing of the form \\\ and then a bracket, or \\\n and then a bracket, remove it
    while re.match(r"\\{2,}\n?\(", s):
        s = s[3:]
    return s


def split_multiletter_symbols(expr):
    reps = {}
    for s in list(expr.free_symbols):
        name = s.name
        if name.isalpha() and len(name) > 1 and not all(ch in "ABCDE" for ch in name):
            reps[s] = Mul(*[Symbol(ch) for ch in name])
    return expr.xreplace(reps)

def check_answers(ans1, ans2):
    """Checks if two answers are equal.

    Args:
        ans1: The first answer.
        ans2: The second answer.

    Returns:
        bool: True if the answers are equal, False otherwise.
    """
    if ans1 is None or ans2 is None:
        return False
    if (type(ans1) in [list, AnswerList]) != (type(ans2) in [list, AnswerList]):
        return False
    try:
        if not (hasattr(ans1, "equals") and callable(ans1.equals)) or not (
            hasattr(ans2, "equals") and callable(ans2.equals)
        ):
            if isinstance(ans1, str) or isinstance(ans2, str):
                # sympy check equality
                return bool(ans1 == ans2)
                        
            # do approximate equal here
            err = abs(N(ans1 - ans2))
            if err < 1e-10 and err / max(abs(N(ans1)), abs(N(ans2))) < 1e-10:
                return True
            return False
        
        if not isinstance(ans1, AnswerList):
            ans1 = split_multiletter_symbols(ans1)
        if not isinstance(ans2, AnswerList):
            ans2 = split_multiletter_symbols(ans2)
        return bool(ans1.equals(ans2))
    except Exception as e:
        logger.warning(f"Error comparing answers {ans1} and {ans2}: {e}")
        return False


class AnswerList:
    """A class for representing a list of answers."""

    def __init__(self, answers: list[Any]):
        """Initializes the AnswerList.

        Args:
            answers (list[Any]): A list of answers.
        """
        if not isinstance(answers, list) and not isinstance(answers, tuple):
            raise ValueError(f"Expected passed answers to be list or tuple, received {type(answers)}")

        valid_answers = []
        for answer in answers:
            if bool(re.search(r"\d", str(answer))):
                valid_answers.append(answer)
            else:
                logger.warning(f"Could not find any numbers in {answer}, removed from list")

        self.answers = list(valid_answers)

    def equals(self, other: list[Any]):
        """Checks if this AnswerList is equal to another list of answers.

        Args:
            other (list[Any]): The other list of answers.

        Returns:
            bool: True if the lists are equal, False otherwise.
        """
        if len(self.answers) != len(other):
            # logger.info(f"Lists {self.answers} and {other} do not have the same length.")
            return False

        match_ids = set()
        for ans1 in self.answers:
            match_found = False
            for i, ans2 in enumerate(other):
                if i not in match_ids and check_answers(ans1, ans2):
                    match_ids.add(i)
                    match_found = True
                    break
            if not match_found:
                # logger.info(f"Could not find a match for element {ans1} in {other}")
                return False
        return True

    def __str__(self):
        return "[" + ",".join([str(ans) for ans in self.answers]) + "]"

    def __len__(self):
        return len(self.answers)

    def __iter__(self):
        return iter(self.answers)


class ParseObject:
    """A base class for parsing objects."""

    @classmethod
    def is_at_start(cls, string):
        """Checks if the object is at the start of a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is at the start of the string, False otherwise.
        """
        return False

    @classmethod
    def is_complete(cls, string):
        """Checks if the object is complete in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is complete in the string, False otherwise.
        """
        return string.count("{") == string.count("}") and string.count("(") == string.count(")")

    @classmethod
    def is_finished(cls, string):
        """Checks if the object is finished in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is finished in the string, False otherwise.
        """
        return True

    @classmethod
    def parse(cls, string):
        """Parses a string into an object.

        Args:
            string (str): The string to parse.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class ParsePrimitive(ParseObject):
    """A class for parsing primitive types."""

    @classmethod
    def parse(cls, string, primitive_type):
        """Parses a string into a primitive type.

        Args:
            string (str): The string to parse.
            primitive_type (type): The primitive type to parse into.

        Returns:
            tuple: A tuple containing the parsed primitive and a warning level.
        """
        warning = WarningType.NONE
        # Integer
        if string.isdigit():
            if primitive_type == Fraction:
                return Fraction(int(string), 1)
            return int(string), warning
        # Float
        try:
            float_string = float(string)
            if int(float_string) == float_string:
                if primitive_type == Fraction:
                    return Fraction(int(float_string), 1)
                return int(float_string), warning
            return float_string, warning
        except ValueError:
            # logger.info(f"Couldn't configure floating point to fraction for {string}")
            pass
        # Expression
        if bool(re.search(r"sqrt(\d+)", string)):
            string = re.sub(r"sqrt(\d+)", r"sqrt{\1}", string)
        if bool(re.search(r"frac(\d)", string)):
            string = re.sub(r"frac(\d)", r"frac{\1}", string)
        try:
            latex_str = string
            for _ in range(5):
                init_str = latex_str
                latex_str = re.sub(r"\\*(?:dfrac|tfrac|frac)\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", latex_str)
                latex_str = re.sub(r"\\*binom\{([^{}]*)\}\{([^{}]*)\}", r"binomial(\1, \2)", latex_str)
                latex_str = re.sub(r"\\*sqrt\[(.*?)\]\{(.*?)\}", r"(\2)**(1/(\1))", latex_str)
                latex_str = re.sub(r"\\*sqrt\((\d+)\)\{([^{}]*)\}", r"(\2)**(1/(\1))", latex_str)
                latex_str = re.sub(r"\\*sqrt\{(.*?)\}", r"(\1)**(1/2)", latex_str)

                latex_str = latex_str.replace("^", "**")
                latex_str = latex_str.replace("\\cdot", "*").replace("\\times", "*")
                latex_str = latex_str.replace("\\pi", " pi ").replace("\\e", " E ").replace("\\i", " I ")
                latex_str = re.sub(r"\bi\b", "I", latex_str)
                if init_str == latex_str:
                    break

            for _ in range(5):

                init_str = latex_str
                latex_str = re.sub(r"\{(\d+)\}", r"(\1)", latex_str)
                latex_str = re.sub(r"\\*(?:dfrac|tfrac|frac)\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", latex_str)
                latex_str = re.sub(r"\\*binom\{([^{}]*)\}\{([^{}]*)\}", r"binomial(\1, \2)", latex_str)
                latex_str = re.sub(r"\\*sqrt\[(.*?)\]\{(.*?)\}", r"(\2)**(1/(\1))", latex_str)
                latex_str = re.sub(r"\\*sqrt\((\d+)\)\{([^{}]*)\}", r"(\2)**(1/(\1))", latex_str)
                latex_str = re.sub(r"\\*sqrt\{(.*?)\}", r"(\1)**(1/2)", latex_str)

                latex_str = latex_str.replace("^", "**")
                latex_str = latex_str.replace("\\cdot", "*").replace("\\times", "*")
                latex_str = latex_str.replace("\\pi", " pi ").replace("\\e", " E ").replace("\\i", " I ")
                latex_str = re.sub(r"\bi\b", "I", latex_str)
                if init_str == latex_str:
                    break

            # Handle implcit multiplication
            latex_str = re.sub(r"(\d|(?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))\(", r"\1*(", latex_str)
            latex_str = re.sub(r"\)(\d|(?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))", r")*\1", latex_str)
            latex_str = re.sub(r"(?<=\d)((?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))", r"*\1", latex_str)
            latex_str = re.sub(r"((?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))(?=\d)", r"\1*", latex_str)
            latex_str = re.sub(r"\{([^{}]*)\}", lambda m: "[" + m.group(1).replace(",", ", ") + "]", latex_str)

            if latex_str == "None":
                string = sympy.core.symbol.Symbol("None")
            else:
                string = sympy.sympify(
                    latex_str,
                    locals={"binomial": sympy.binomial, "pi": sympy.pi, "E": sympy.E, "e": sympy.E, "I": sympy.I},
                )
        except Exception as e:
            try:
                string_no_eq = string
                if "=" in string_no_eq:
                    # rfind is used to remove the last occurence of "="
                    string_no_eq = string_no_eq[string_no_eq.rfind("=") + 1 :]
                output_val = latex2sympy_fixed(string_no_eq)
                # print complex and real part separately

                try:
                    simplified = simplify(output_val)
                    if simplified.is_integer:
                        return int(simplified), warning
                    if isinstance(simplified, sympy.Rational):
                        return simplified, warning
                    float_val = float(N(output_val, 101))
                    if float("inf") == float_val or float("-inf") == float_val:
                        return int(N(latex2sympy_fixed(string_no_eq), 50001)), warning  # important for large ints
                    return float_val, warning
                except:  # noqa: E722
                    try:
                        complex_val = complex(N(output_val, 101))
                        return complex_val, warning
                    except:  # noqa: E722
                        return output_val, warning
            except Exception as e:
                logger.warning(f"Error: Custom parsing error {e}, {string_no_eq}")
                warning = max(warning, WarningType.MAJOR)
                return None, warning

        return string, warning

    @classmethod
    def is_at_start(cls, string):
        return True


class ParseList(ParseObject):
    """A class for parsing lists."""

    @classmethod
    def is_at_start(cls, string):
        """Checks if the object is at the start of a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is at the start of the string, False otherwise.
        """
        return string.startswith(r"(")

    @classmethod
    def is_finished(cls, string):
        """Checks if the object is finished in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is finished in the string, False otherwise.
        """
        # safe condition for finishing a list
        return string.strip().strip(",").endswith(")")

    @classmethod
    def is_complete(cls, string):
        """Checks if the object is complete in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is complete in the string, False otherwise.
        """
        return string.count("(") == string.count(")")

    @classmethod
    def never_zero_count(cls, string):
        """Checks if the parenthesis count never reaches zero before the end of the string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the parenthesis count never reaches zero, False otherwise.
        """
        # says wheter count "(" - count ")" for every string[:i] is never zero
        count = 0
        ever_zero = False
        for char in string:
            if char == "(":
                count += 1
            if char == ")":
                count -= 1
            if count == 0:
                ever_zero = True
        return not ever_zero

    @classmethod
    def parse(cls, string, delimiter=[r"\n", ","], primitive_type=None, depth=0):
        """Parses a string into a list.

        Args:
            string (str): The string to parse.
            delimiter (list[str], optional): The delimiter to use. Defaults to [r"\n", ","].
            primitive_type (type, optional): The primitive type to parse into. Defaults to None.
            depth (int, optional): The recursion depth. Defaults to 0.

        Returns:
            tuple: A tuple containing the parsed list and a warning level.
        """
        if isinstance(delimiter, str):
            delimiter = [delimiter]
        output = []
        if not string.startswith("("):
            return None
        string = string.strip().strip(",")
        if cls.never_zero_count(string[:-1]):
            string = string[1:-1]
        string = strip(string)
        used_delim = delimiter[0]
        for delim in delimiter:
            if delim in string:
                comma_separated = string.split(delim)
                used_delim = delim
                break
        warning = WarningType.NONE
        while len(string) > 0:
            previous_string = string
            comma_separated = string.split(used_delim)
            allowed_objects = [ParseList, ParsePrimitive]
            if depth > 50:
                allowed_objects = [ParsePrimitive]
            for obj in allowed_objects:
                if obj.is_at_start(strip(string)):
                    current_index = 1
                    while not obj.is_complete(
                        strip(used_delim.join(comma_separated[:current_index]))
                    ) or not obj.is_finished(strip(used_delim.join(comma_separated[:current_index]))):
                        current_index += 1
                        if current_index >= len(comma_separated):
                            break
                    if not obj.is_complete(
                        strip(used_delim.join(comma_separated[:current_index]))
                    ) or not obj.is_finished(strip(used_delim.join(comma_separated[:current_index]))):
                        continue

                    if obj == ParseList:
                        parsed, new_warning = obj.parse(
                            strip(used_delim.join(comma_separated[:current_index])),
                            primitive_type=primitive_type,
                            depth=depth + 1,
                        )
                    else:
                        parsed, new_warning = obj.parse(
                            strip(used_delim.join(comma_separated[:current_index])), primitive_type=primitive_type
                        )
                    warning = max(warning, new_warning)
                    output.append(parsed)
                    string = strip(used_delim.join(comma_separated[current_index:]))
                    break
            if previous_string == string:
                if depth > 50:
                    logger.error(f"Response {string} reached depth > 50")
                    raise ValueError(f"Failed to parse '{string}'")
                return None, WarningType.MAJOR
        return output, warning
