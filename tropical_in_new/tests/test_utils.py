import pytest

from tropical_in_new.src.utils import read_evidence_file, read_model_file, read_model_from_string


def test_read_evidence_file(tmp_path):
    content = "2 0 1 3 0\n"
    filepath = tmp_path / "example.evid"
    filepath.write_text(content, encoding="utf-8")

    evidence = read_evidence_file(str(filepath))
    assert evidence == {1: 1, 4: 0}


def test_read_evidence_file_empty_path():
    assert read_evidence_file("") == {}


def test_read_evidence_file_empty_content(tmp_path):
    filepath = tmp_path / "empty.evid"
    filepath.write_text("", encoding="utf-8")
    assert read_evidence_file(str(filepath)) == {}


def test_read_evidence_file_malformed(tmp_path):
    filepath = tmp_path / "bad.evid"
    filepath.write_text("2 0 1\n", encoding="utf-8")  # declares 2 obs vars but only 1 pair
    with pytest.raises(ValueError, match="Malformed evidence line"):
        read_evidence_file(str(filepath))


def test_read_model_from_string_malformed_header():
    with pytest.raises(ValueError, match="at least 4 header lines"):
        read_model_from_string("MARKOV\n2\n")


def test_read_model_from_string_bad_network_type():
    with pytest.raises(ValueError, match="Unsupported UAI network type"):
        read_model_from_string("UNKNOWN\n2\n2 2\n1\n1 0\n2\n0.5 0.5\n")


def test_read_model_from_string_card_mismatch():
    with pytest.raises(ValueError, match="Expected 2 cardinalities"):
        read_model_from_string("MARKOV\n2\n2 2 2\n1\n1 0\n2\n0.5 0.5\n")


def test_read_model_from_string_scope_size_mismatch():
    with pytest.raises(ValueError, match="Scope size mismatch"):
        read_model_from_string("MARKOV\n2\n2 2\n1\n2 0\n2\n0.5 0.5\n")


def test_read_model_from_string_table_size_mismatch():
    with pytest.raises(ValueError, match="Factor table size mismatch"):
        read_model_from_string("MARKOV\n2\n2 2\n1\n2 0 1\n3\n0.5 0.5 0.5\n")


def test_read_model_file(tmp_path):
    content = "MARKOV\n2\n2 2\n1\n1 0\n2\n0.6 0.4\n"
    filepath = tmp_path / "test.uai"
    filepath.write_text(content, encoding="utf-8")
    model = read_model_file(str(filepath))
    assert model.nvars == 2
    assert len(model.factors) == 1
