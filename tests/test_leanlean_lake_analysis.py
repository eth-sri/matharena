import pytest

from leanlean.lake_analysis import aggregate, operation, union_seconds


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["lake", "-q", "build", "Solution"], "build"),
        (["lake", "+v4.35.0", "--dir", "/work", "comparator"], "comparator"),
        (["lake", "comparator", "--help"], "help"),
        (["lake", "env", "lean", "Solution.lean"], "lean_check"),
        (["lake", "env", "python", "analyze.py"], "other"),
    ],
)
def test_identifies_real_checks_separately_from_help(arguments, expected):
    assert operation(arguments) == expected


def test_overlap_is_not_double_counted_and_interruption_is_not_measured():
    def completed(start, end, code):
        return {
            "status": "complete",
            "execution_start_ns": start * 10**9,
            "end_monotonic_ns": end * 10**9,
            "duration_s": end - start,
            "returncode": code,
        }

    rows = [
        completed(1, 5, 0),
        completed(2, 3, 1),
        completed(4, 7, -15),
        {"status": "missing_end", "duration_upper_bound_s": 100},
    ]
    result = aggregate(rows)
    assert result["process_seconds"] == 8
    assert result["busy_wall_seconds"] == 6
    assert result["incomplete"] == 1
    assert result["nonzero_exit"] == 2
    assert result["succeeded"] == 1
    assert result["median_s"] == 3


def test_union_rejects_invalid_clock_order():
    with pytest.raises(ValueError):
        union_seconds([(4, 3)])
