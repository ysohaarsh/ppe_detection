import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PPEDetection"))

import PPEDetection as ppe  # noqa: E402


def test_missing_ppe_is_red():
    for name in ("NO-Hardhat", "NO-Mask", "NO-Safety Vest"):
        assert ppe.color_for_class(name) == ppe.RED


def test_present_ppe_is_green():
    for name in ("Hardhat", "Mask", "Safety Vest"):
        assert ppe.color_for_class(name) == ppe.GREEN


def test_other_classes_are_blue():
    for name in ("Person", "vehicle", "machinery", "Safety Cone"):
        assert ppe.color_for_class(name) == ppe.BLUE


def test_round_up_confidence():
    assert ppe.round_up_confidence(0.731) == 0.74
    assert ppe.round_up_confidence(0.5) == 0.5


def test_parse_args_defaults():
    args = ppe.parse_args([])
    assert args.source == "0" and args.conf == 0.5
    assert args.model.endswith("ppe.pt")


def test_class_names_cover_model_outputs():
    assert len(ppe.CLASS_NAMES) == 10
