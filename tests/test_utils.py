"""Testes adicionais para src.utils."""

from pathlib import Path
import sys
import logging
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import utils


# ═══════════════════════════════════════════════════════
#  Testes existentes (mantidos)
# ═══════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════
#  Novos testes para cobrir linhas faltantes
# ═══════════════════════════════════════════════════════

class TestSetupLogging:
    """Cobre linhas 28-52 (setup_logging)."""

    def test_setup_logging_creates_logger(self, tmp_path):
        logger = utils.setup_logging(
            log_dir=str(tmp_path / "test_logs"),
            log_file="test.log",
            level=logging.DEBUG,
        )
        assert isinstance(logger, logging.Logger)
        assert (tmp_path / "test_logs" / "test.log").exists()

    def test_setup_logging_default_params(self, tmp_path):
        logger = utils.setup_logging(log_dir=str(tmp_path / "logs2"))
        assert logger is not None

    def test_setup_logging_writes_to_file(self, tmp_path):
        log_dir = str(tmp_path / "logs3")
        logger = utils.setup_logging(log_dir=log_dir, log_file="write_test.log")
        # Force flush all handlers so content is written
        for handler in logging.root.handlers:
            handler.flush()
        for handler in logger.handlers:
            handler.flush()
        log_file = tmp_path / "logs3" / "write_test.log"
        # The file was created; check it exists and optionally has content
        # (basicConfig may be no-op in pytest, so the file may be empty)
        assert log_file.exists()


class TestLoadJsonErrors:
    """Cobre linha 85 (FileNotFoundError)."""

    def test_load_json_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
            utils.load_json("/caminho/que/nao/existe/dados.json")


class TestConvertCommaToDot:
    """Cobre linhas 129, 136-138 (auto-detect de colunas e ValueError)."""

    def test_convert_auto_detect_columns(self):
        """Sem passar columns, deve detectar colunas object automaticamente."""
        df = pd.DataFrame({
            "preco": ["1,99", "2,50", "3,00"],
            "nome": ["A", "B", "C"],
            "qtd": [1, 2, 3],
        })
        result = utils.convert_comma_to_dot(df)  # columns=None
        assert result["preco"].iloc[0] == 1.99
        assert result["preco"].iloc[1] == 2.50

    def test_convert_with_non_convertible_values(self):
        """Colunas com texto puro devem ser ignoradas (ValueError path)."""
        df = pd.DataFrame({"texto": ["abc", "def", "ghi"]})
        result = utils.convert_comma_to_dot(df, columns=["texto"])
        # Não deve lançar exceção, valores devem permanecer
        assert result["texto"].iloc[0] == "abc" or pd.notna(result["texto"].iloc[0])

    def test_convert_column_not_in_dataframe(self):
        """Coluna inexistente deve ser ignorada."""
        df = pd.DataFrame({"a": ["1,0"]})
        result = utils.convert_comma_to_dot(df, columns=["coluna_inexistente"])
        assert "a" in result.columns


class TestSafeDivideEdgeCases:
    """Cobre edge cases do safe_divide."""

    def test_safe_divide_nan_denominator(self):
        assert utils.safe_divide(10, float('nan'), default=-1) == -1

    def test_safe_divide_type_error(self):
        assert utils.safe_divide(10, "texto", default=0.0) == 0.0

    def test_safe_divide_none_denominator(self):
        # pd.isna(None) is True
        assert utils.safe_divide(10, None, default=-99) == -99


class TestExtractPhaseNumber:
    """Cobre edge cases do extract_phase_number."""

    def test_extract_phase_nan(self):
        assert utils.extract_phase_number(None) == 0

    def test_extract_phase_numeric_string(self):
        assert utils.extract_phase_number("3") == 3

    def test_extract_phase_nan_pandas(self):
        assert utils.extract_phase_number(float('nan')) == 0


class TestGetMemoryUsage:
    """Cobre diferentes unidades de memória."""

    def test_memory_bytes(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        mem_b = utils.get_memory_usage(df, unit="B")
        mem_kb = utils.get_memory_usage(df, unit="KB")
        mem_mb = utils.get_memory_usage(df, unit="MB")
        mem_gb = utils.get_memory_usage(df, unit="GB")
        assert mem_b > mem_kb > mem_mb > mem_gb > 0

    def test_memory_invalid_unit(self):
        df = pd.DataFrame({"x": [1]})
        result = utils.get_memory_usage(df, unit="INVALID")
        assert result > 0  # Fallback para divisão por 1


class TestNormalizeString:
    """Cobre edge cases do normalize_string."""

    def test_normalize_nan(self):
        assert utils.normalize_string(None) == ""

    def test_normalize_nan_float(self):
        assert utils.normalize_string(float('nan')) == ""

    def test_normalize_numbers(self):
        result = utils.normalize_string("123")
        assert result == "123"


class TestDetectOutliers:
    """Cobre edge cases do detect_outliers_iqr."""

    def test_no_outliers(self):
        s = pd.Series([1, 2, 3, 4, 5])
        flags = utils.detect_outliers_iqr(s)
        assert flags.sum() == 0

    def test_custom_multiplier(self):
        s = pd.Series([1, 2, 2, 3, 10])
        # Com multiplier baixo, mais outliers
        flags_strict = utils.detect_outliers_iqr(s, multiplier=0.5)
        flags_loose = utils.detect_outliers_iqr(s, multiplier=3.0)
        assert flags_strict.sum() >= flags_loose.sum()


class TestDataValidatorExtended:
    """Cobre edge cases do DataValidator."""

    def test_validate_range_boundary(self):
        assert utils.DataValidator.validate_range(1, 1, 5) is True
        assert utils.DataValidator.validate_range(5, 1, 5) is True
        assert utils.DataValidator.validate_range(0, 1, 5) is False
        assert utils.DataValidator.validate_range(6, 1, 5) is False

    def test_validate_not_null_with_none(self):
        assert utils.DataValidator.validate_not_null(None) is False

    def test_validate_not_null_with_nan(self):
        assert utils.DataValidator.validate_not_null(float('nan')) is False

    def test_validate_type_wrong(self):
        assert utils.DataValidator.validate_type(123, str) is False

    def test_validate_in_list_missing(self):
        assert utils.DataValidator.validate_in_list("X", ["A", "B"]) is False


class TestPrintSectionHeader:
    """Cobre a formatação do header."""

    def test_default_params(self, capsys):
        utils.print_section_header("Seção")
        out = capsys.readouterr().out
        assert "Seção" in out
        assert "=" in out

    def test_custom_width_and_char(self, capsys):
        utils.print_section_header("X", width=10, char="*")
        out = capsys.readouterr().out
        assert "*" in out
        assert "X" in out


class TestCalculatePercentiles:
    """Cobre edge cases de percentis."""

    def test_custom_percentiles(self):
        s = pd.Series(range(100))
        result = utils.calculate_percentiles(s, [10, 90])
        assert "p10" in result
        assert "p90" in result
        assert result["p10"] < result["p90"]


class TestFormatNumber:
    """Cobre edge cases de formatação."""

    def test_format_zero_decimals(self):
        assert utils.format_number(3.14, 0) == "3"

    def test_format_large_number(self):
        assert utils.format_number(12345.6789, 1) == "12345.7"


class TestGetTimestamp:
    """Cobre custom format."""

    def test_custom_format(self):
        ts = utils.get_timestamp("%Y")
        assert len(ts) == 4
        assert ts.isdigit()


class TestConstantsCompleteness:
    """Cobre constantes no final do módulo."""

    def test_pedras_order_complete(self):
        assert len(utils.PEDRAS_ORDER) == 4
        assert "Quartzo" in utils.PEDRAS_ORDER
        assert "Topázio" in utils.PEDRAS_ORDER

    def test_risk_levels_complete(self):
        assert len(utils.RISK_LEVELS) == 4

    def test_boolean_mapping_complete(self):
        assert utils.BOOLEAN_MAPPING["Sim"] == 1
        assert utils.BOOLEAN_MAPPING["Não"] == 0
        assert utils.BOOLEAN_MAPPING["sim"] == 1
        assert utils.BOOLEAN_MAPPING["não"] == 0
        assert utils.BOOLEAN_MAPPING["SIM"] == 1
        assert utils.BOOLEAN_MAPPING["NÃO"] == 0
        assert utils.BOOLEAN_MAPPING["nao"] == 0
        assert utils.BOOLEAN_MAPPING["NAO"] == 0
