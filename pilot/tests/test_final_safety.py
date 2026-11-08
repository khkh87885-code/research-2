import pytest

from compression_study.statistics import summarize
from compression_study.dashboard import build_dashboard


def test_missing_patient_is_never_a_patient_named_none():
    with pytest.raises(ValueError, match="known patient"):
        summarize([dict(dataset="x", split="test", prompt_strategy="x", patient_key=None,
                        image_key="a", condition="baseline", status="ok", dice=.8)])


def test_dashboard_direct_api_does_not_overwrite(tmp_path):
    target = tmp_path / "existing.html"
    target.write_text("original")
    with pytest.raises(FileExistsError):
        build_dashboard([], target)
    assert target.read_text() == "original"
