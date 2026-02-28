"""Testes adicionais para src.utils."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import utils


def test_json_and_dataframe_validation(tmp_path):
    data = {"k": 1, "nested": {"a": True}}
    file_path = tmp_path / "a" / "b" / "data.json"

    utils.save_json(data, file_path)
    loaded = utils.load_json(file_path)

    assert loaded == data

    df = pd.DataFrame({"c1": [1, 2], "c2": [3, 4]})
    assert utils.validate_dataframe(df, ["c1"]) is True
    assert utils.validate_dataframe(df, ["missing"]) is False


def test_convert_percentiles_timestamp_and_formatting():
    df = pd.DataFrame({"num": ["1,5", "2,0", None], "txt": ["x", "y", "z"]})
    conv = utils.convert_comma_to_dot(df, columns=["num"])
    assert conv["num"].iloc[0] == 1.5

    pct = utils.calculate_percentiles(pd.Series([1, 2, 3, 4]))
    assert pct["p50"] == 2.5

    formatted = utils.format_number(3.14159, 3)
    assert formatted == "3.142"

    ts = utils.get_timestamp()
    assert isinstance(ts, str) and len(ts) >= 8


def test_directories_division_and_phase_extraction(tmp_path):
    p1 = tmp_path / "x"
    p2 = tmp_path / "y" / "z"
    utils.create_directories(p1, p2)

    assert p1.exists()
    assert p2.exists()

    assert utils.safe_divide(10, 2) == 5
    assert utils.safe_divide(10, 0, default=-1) == -1

    assert utils.extract_phase_number("Fase 7 (Ensino Médio)") == 7
    assert utils.extract_phase_number("sem fase") == 0


def test_memory_outliers_normalize_ci_and_header(capsys):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    mem = utils.get_memory_usage(df, unit="KB")
    assert mem > 0

    flags = utils.detect_outliers_iqr(pd.Series([1, 2, 2, 3, 100]))
    assert bool(flags.iloc[-1]) is True

    assert utils.normalize_string(" ÁGATA ") == "agata"
    assert utils.normalize_string(" ÁGATA ") == "agata"

    low, high = utils.calculate_confidence_interval(np.array([1, 2, 3, 4, 5]))
    assert low < high

    utils.print_section_header("Teste", width=20, char="-")
    out = capsys.readouterr().out
    assert "Teste" in out


def test_datavalidator_and_constants():
    assert utils.DataValidator.validate_range(3, 1, 5)
    assert utils.DataValidator.validate_not_null(1)
    assert utils.DataValidator.validate_type("abc", str)
    assert utils.DataValidator.validate_in_list("Sim", utils.BOOLEAN_VALUES)

    assert utils.PEDRAS_MAPPING["Quartzo"] == 1
    assert "Crítico" in utils.RISK_LEVELS
    assert utils.BOOLEAN_MAPPING["Sim"] == 1
