import pytest

from iacs_v2.statistics import paired_difference, bootstrap_ci, holm, _case_means


def row(case, image, value, status="ok"):
    return dict(case_key=case, image_key=image, dice=value, status=status)


def test_case_pairing_averages_images_before_cases():
    first = [row("a", "a1", 0), row("a", "a2", 2), row("b", "b1", 100)]
    second = [row("a", "a1", 0), row("a", "a2", 0), row("b", "b1", 0)]
    result = paired_difference(first, second, "dice", 100)
    assert result["valid_paired_cases"] == 2
    assert result["paired_images"] == 3
    assert result["mean_difference"] == pytest.approx(50.5)


def test_failed_records_retained_in_pair_denominator():
    first = [row("a", "a1", .5), row("b", "b1", None, "error")]
    second = [row("a", "a1", .6), row("b", "b1", .7)]
    result = paired_difference(first, second, "dice", 100)
    assert result["scheduled_cases"] == 2 and result["valid_paired_cases"] == 1
    assert result["missing_or_failed_cases"] == 1 and result["ci95"] is None
    assert result["clinical_noninferiority"] is None


def test_holm_and_bootstrap_known_values():
    rows = [dict(p_value=.01), dict(p_value=.04), dict(p_value=.03), dict(p_value=None)]
    result = holm(rows)
    assert [r["holm_p"] for r in result[:3]] == pytest.approx([.03, .06, .06])
    assert result[3]["holm_p"] is None
    assert bootstrap_ci([2, 2, 2], 100) == [2., 2.]
    assert bootstrap_ci([1], 100) is None


def test_missing_case_not_treated_as_individual_and_duplicate_rejected():
    with pytest.raises(ValueError):
        _case_means([row(None, "x", .8)], "dice")
    with pytest.raises(ValueError):
        paired_difference([row("a", "x", .8), row("a", "x", .9)], [], "dice", 100)
