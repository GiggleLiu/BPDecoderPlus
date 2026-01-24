from tropical_in_new.src.utils import read_evidence_file


def test_read_evidence_file(tmp_path):
    content = "2 0 1 3 0\n"
    filepath = tmp_path / "example.evid"
    filepath.write_text(content, encoding="utf-8")

    evidence = read_evidence_file(str(filepath))
    assert evidence == {1: 1, 4: 0}
