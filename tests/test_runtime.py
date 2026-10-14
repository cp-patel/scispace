import logging

from research_assistant.runtime import load_calibration


def test_missing_calibration_file_is_reported(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert load_calibration(str(tmp_path / "missing.json")) is None
    assert "calibration" in caplog.text
