"""Lean source-token metric using the Lean Refactor paper's small lexer.

The paper's original proof-length metric strips a theorem statement and counts
only its proof body.  ``proof_length`` retains that reference behavior.

For repository compression, however, every non-comment, non-import source token
is counted.  This includes theorem statements and helper declarations, so
splitting a proof into many small lemmas does not make declaration overhead
free.  The module is also a standalone script copied into benchmark containers.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path


PROOF_LENGTH_ERROR = 10**9

# Operators rejoined after the character-level lexer, matching the tokenizer
# published in the Lean Refactor appendix.
LEAN_OPERATORS = (
    ":=",
    "!=",
    "&&",
    "-.",
    "->",
    "←",
    "..",
    "...",
    "::",
    ":>",
    "<;>",
    ";;",
    "==",
    "||",
    "=>",
    "<=",
    ">=",
    "−1",
    "?_",
)
_SPACED_OPERATORS = tuple(" ".join(operator) for operator in LEAN_OPERATORS)
_SPACED_OPERATOR_MAP = dict(zip(_SPACED_OPERATORS, LEAN_OPERATORS, strict=False))

_PROOF_COMMANDS = {"theorem", "lemma", "example"}
_IMPORT_COMMAND_RE = re.compile(r"^\s*(?:(?:public|private)\s+)?import(?:\s|$)")
# One module name after `import`; command keywords and punctuation end the list.
_IMPORT_MODULE_RE = re.compile(r"[^\s()\[\]{}:,;@#\"'`]+")
_IMPORT_LIST_END = frozenset(
    """
    theorem lemma example def abbrev opaque axiom constant structure class inductive
    instance namespace section end variable universe include omit open export attribute
    initialize macro macro_rules syntax elab elab_rules command_elab set_option mutual
    notation infix infixl infixr prefix postfix declare_syntax_cat deriving
    private protected noncomputable unsafe partial local scoped
    """.split()
)

_COMMAND_START_RE = re.compile(
    r"""
    ^[ \t]*
    (?:@\[[^\n]*\][ \t]*)*
    (?:(?:private|protected|noncomputable|unsafe|partial|local|scoped)[ \t]+)*
    (?P<kind>
        theorem|lemma|example|def|abbrev|opaque|axiom|constant|
        structure|class|inductive|instance|namespace|section|end|
        variable|include|omit|open|export|attribute|initialize|
        macro|syntax|elab|command_elab|set_option|mutual
    )\b
    """,
    re.VERBOSE,
)


def remove_lean_comments(source: str, *, mask_strings: bool = False) -> str:
    """Mask comments while preserving offsets; optionally mask strings too."""

    result: list[str] = []
    index = 0
    block_depth = 0
    in_line_comment = False
    in_string = False

    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                result.append(char)
            else:
                result.append(" ")
            index += 1
            continue

        if block_depth:
            if char == "/" and following == "-":
                result.extend((" ", " "))
                block_depth += 1
                index += 2
                continue
            if char == "-" and following == "/":
                result.extend((" ", " "))
                block_depth -= 1
                index += 2
                continue
            result.append(char if char == "\n" else " ")
            index += 1
            continue

        if in_string:
            result.append(" " if mask_strings and char != "\n" else char)
            if char == "\\" and following:
                result.append(" " if mask_strings and following != "\n" else following)
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue

        if char == "-" and following == "-":
            result.extend((" ", " "))
            in_line_comment = True
            index += 2
            continue
        if char == "/" and following == "-":
            result.extend((" ", " "))
            block_depth = 1
            index += 2
            continue
        if char == '"':
            in_string = True

        result.append(" " if mask_strings and char == '"' else char)
        index += 1

    return "".join(result)


def lean_refactor_lexer(lean_snippet: str) -> str:
    """Tokenize a Lean snippet with the lexer from the paper appendix."""

    tokenized_lines: list[str] = []
    for line in lean_snippet.splitlines():
        tokens: list[str] = []
        token = ""
        for char in line:
            if char == " ":
                if token:
                    tokens.append(token)
                token = ""
            elif char.isalnum() or char in "._'":
                token += char
            else:
                if token:
                    tokens.append(token)
                token = ""
                tokens.append(char)
        if token:
            tokens.append(token)

        tokenized_line = " ".join(tokens)
        for spaced_operator in _SPACED_OPERATORS:
            if spaced_operator in tokenized_line:
                tokenized_line = tokenized_line.replace(
                    spaced_operator,
                    _SPACED_OPERATOR_MAP[spaced_operator],
                )
        tokenized_lines.append(tokenized_line)

    return "\n".join(tokenized_lines)


def _token_count(proof: str) -> int:
    tokenized = lean_refactor_lexer(proof.strip())
    # The appendix uses ``split(" ")``, which accidentally counts an empty
    # tokenized line as one token.  Ignore empty fields so whitespace-only lines
    # cannot change a repository compression score.
    return sum(len(line.split()) for line in tokenized.splitlines())


def _equation_proof_bodies(statement_and_proof: str) -> list[str]:
    """Extract RHS proofs from a Lean equation-style theorem declaration."""

    masked = remove_lean_comments(statement_and_proof, mask_strings=True)
    source = remove_lean_comments(statement_and_proof)
    clauses: list[tuple[int, int]] = []
    offset = 0
    equation_indent: int | None = None
    clause_re = re.compile(r"^(?P<indent>[ \t]*)\|.*?=>")
    for line in masked.splitlines(keepends=True):
        match = clause_re.match(line)
        if match:
            indentation = len(match.group("indent"))
            if equation_indent is None:
                equation_indent = indentation
            if indentation == equation_indent:
                clauses.append((offset, offset + match.end()))
        offset += len(line)

    bodies: list[str] = []
    for index, (clause_start, proof_start) in enumerate(clauses):
        proof_end = clauses[index + 1][0] if index + 1 < len(clauses) else len(source)
        body = source[proof_start:proof_end].strip()
        if body == "by":
            body = ""
        elif body.startswith("by") and len(body) > 2 and body[2].isspace():
            body = body[2:].strip()
        bodies.append(body)
    return bodies


def proof_length(statement_and_proof: str) -> int:
    """Return the paper-style token count of one theorem/lemma/example proof."""

    try:
        without_comments = remove_lean_comments(statement_and_proof)
        if ":= by" in without_comments:
            proof = without_comments.split(":= by", maxsplit=1)[1].strip()
            return _token_count(proof)
        if ":=" in without_comments:
            proof = without_comments.split(":=", maxsplit=1)[1].strip()
            return _token_count(proof)
        equation_bodies = _equation_proof_bodies(statement_and_proof)
        if not equation_bodies:
            return PROOF_LENGTH_ERROR
        return sum(_token_count(body) for body in equation_bodies)
    except Exception:
        return PROOF_LENGTH_ERROR



def split_import_command(source_line: str, masked_line: str) -> tuple[list[str], str]:
    """Split a line into the module names it imports and the code after them.

    Lean accepts commands after the imports on the same line
    (``import Lean theorem t : True := trivial``), so only the import command
    itself is unscored. ``masked_line`` is the comment- and string-masked form
    of ``source_line`` (same offsets). A line that is not an import is all code.
    """

    match = _IMPORT_COMMAND_RE.match(masked_line)
    if not match:
        return [], source_line
    modules = []
    position = match.end()
    while True:
        start = position + len(masked_line[position:]) - len(masked_line[position:].lstrip())
        token = _IMPORT_MODULE_RE.match(masked_line, start)
        if token is None or token.group() in _IMPORT_LIST_END:
            return modules, source_line[start:]
        modules.append(token.group())
        position = token.end()


def count_lean_tokens_in_source(source: str) -> int:
    """Count all source tokens except comments and import commands."""

    without_comments = remove_lean_comments(source)
    masked_strings = remove_lean_comments(source, mask_strings=True)
    without_imports = "\n".join(
        split_import_command(source_line, masked_line)[1]
        for source_line, masked_line in zip(
            without_comments.splitlines(),
            masked_strings.splitlines(),
            strict=True,
        )
    )
    return _token_count(without_imports)


def _iter_metric_files(
    root: Path,
    exclude_dirs: list[str],
    include_prefix: str,
    exclude_files: set[str] | None = None,
) -> Iterator[Path]:
    """Yield the same scoped Lean file set as the benchmark word metric."""

    root = root.resolve()
    normalized_excludes = [item.strip().strip("/") for item in exclude_dirs if item.strip()]
    include_prefix = include_prefix.strip().strip("/")

    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)
        relative_root = current.relative_to(root)
        relative_root_text = "" if relative_root == Path(".") else relative_root.as_posix()

        if ".lake" in relative_root.parts:
            dirs[:] = []
            continue
        if any(
            relative_root_text == excluded
            or relative_root_text.startswith(excluded + "/")
            for excluded in normalized_excludes
        ):
            dirs[:] = []
            continue
        if include_prefix and relative_root_text and not (
            relative_root_text == include_prefix
            or relative_root_text.startswith(include_prefix + "/")
            or include_prefix.startswith(relative_root_text + "/")
        ):
            dirs[:] = []
            continue

        for filename in files:
            if not filename.endswith(".lean") or filename == "lakefile.lean":
                continue
            path = current / filename
            relative_file = path.relative_to(root).as_posix()
            if relative_file in (exclude_files or set()):
                continue
            if include_prefix:
                if not (
                    relative_file == include_prefix + ".lean"
                    or relative_file.startswith(include_prefix + "/")
                ):
                    continue
            yield path


def measure_repository_lean_tokens(
    root: Path,
    exclude_dirs: list[str] | None = None,
    include_prefix: str = "",
    exclude_files: list[str] | None = None,
) -> int:
    """Return aggregate non-comment, non-import Lean source tokens."""

    total_tokens = 0
    normalized_excluded_files = {
        path.strip().strip("/") for path in (exclude_files or []) if path.strip()
    }
    for path in _iter_metric_files(
        root,
        exclude_dirs or [],
        include_prefix,
        normalized_excluded_files,
    ):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        total_tokens += count_lean_tokens_in_source(source)
    return total_tokens


def _metric_git_path_in_scope(
    path: str, exclude_dirs: list[str], include_prefix: str
) -> bool:
    """Mirror ``_iter_metric_files`` for a path stored in a Git tree."""

    path = path.strip().lstrip("/")
    if (
        not path.endswith(".lean")
        or path.endswith("/lakefile.lean")
        or path == "lakefile.lean"
    ):
        return False
    parts = Path(path).parts
    if ".lake" in parts:
        return False
    normalized_excludes = [
        item.strip().strip("/") for item in exclude_dirs if item.strip()
    ]
    if any(
        path == excluded or path.startswith(excluded + "/")
        for excluded in normalized_excludes
    ):
        return False
    include_prefix = include_prefix.strip().strip("/")
    if include_prefix and not (
        path == include_prefix + ".lean" or path.startswith(include_prefix + "/")
    ):
        return False
    return True


def measure_git_tree_lean_tokens(
    root: Path,
    treeish: str,
    exclude_dirs: list[str] | None = None,
    include_prefix: str = "",
) -> int:
    """Count tokens from one immutable Git tree, independent of live edits."""

    root = root.resolve()
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-tree",
            "-r",
            "-z",
            "--format=%(objectname)%x09%(path)",
            treeish,
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    entries: list[tuple[str, str]] = []
    for raw_entry in listing.split(b"\0"):
        if not raw_entry:
            continue
        raw_oid, raw_path = raw_entry.split(b"\t", 1)
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if _metric_git_path_in_scope(path, exclude_dirs or [], include_prefix):
            entries.append((raw_oid.decode("ascii"), path))
    if not entries:
        return 0

    batch = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=("\n".join(oid for oid, _ in entries) + "\n").encode("ascii"),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    offset = 0
    total_tokens = 0
    for _oid, _path in entries:
        header_end = batch.index(b"\n", offset)
        header = batch[offset:header_end].decode("ascii")
        size = int(header.rsplit(" ", 1)[1])
        start = header_end + 1
        source = batch[start : start + size].decode("utf-8", errors="replace")
        total_tokens += count_lean_tokens_in_source(source)
        offset = start + size + 1
    return total_tokens


def main() -> None:
    root = Path(os.environ.get("LEAN_TOKEN_ROOT", "/testbed"))
    include_prefix = os.environ.get("LEAN_TOKEN_INCLUDE", "")
    treeish = os.environ.get("LEAN_TOKEN_GIT_TREE", "")
    if treeish:
        tokens = measure_git_tree_lean_tokens(
            root,
            treeish,
            exclude_dirs=sys.argv[1:],
            include_prefix=include_prefix,
        )
    else:
        tokens = measure_repository_lean_tokens(
            root,
            exclude_dirs=sys.argv[1:],
            include_prefix=include_prefix,
        )
    print(tokens)


if __name__ == "__main__":
    main()
