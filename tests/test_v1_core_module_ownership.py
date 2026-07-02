from pathlib import Path


def test_config_is_core_owned_and_vision_config_source_is_deleted() -> None:
    core_text = Path("snookerhelp/core/config.py").read_text(encoding="utf-8")

    assert "def resolve_path" in core_text
    assert "def save_yaml" in core_text
    assert not Path("vision/config.py").exists()


def test_table_model_is_core_owned_and_vision_table_model_source_is_deleted() -> None:
    core_text = Path("snookerhelp/core/table.py").read_text(encoding="utf-8")

    assert "class TableModel" in core_text
    assert not Path("vision/table_model.py").exists()


def test_table_warp_is_calibration_owned_and_vision_table_warp_source_is_deleted() -> None:
    calibration_text = Path("snookerhelp/calibration/homography_bootstrap.py").read_text(
        encoding="utf-8"
    )

    assert "class TableWarp" in calibration_text
    assert not Path("vision/table_warp.py").exists()
