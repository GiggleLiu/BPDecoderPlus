import tempfile
import unittest
import torch

try:
    from ._path import add_project_root_to_path
except ImportError:
    from _path import add_project_root_to_path

add_project_root_to_path()

from bpdecoderplus.pytorch_bp import read_model_from_string, read_evidence_file
from bpdecoderplus.pytorch_bp.uai_parser import Factor, UAIModel


class TestUAIParser(unittest.TestCase):
    def test_read_model_from_string(self):
        content = "\n".join(
            [
                "MARKOV",
                "2",
                "2 2",
                "2",
                "1 0",
                "2 0 1",
                "2",
                "0.6 0.4",
                "4",
                "0.9 0.1 0.2 0.8",
            ]
        )
        model = read_model_from_string(content)
        self.assertEqual(model.nvars, 2)
        self.assertEqual(model.cards, [2, 2])
        self.assertEqual(len(model.factors), 2)

        factor0 = model.factors[0]
        factor1 = model.factors[1]
        self.assertEqual(factor0.vars, (1,))
        self.assertEqual(factor1.vars, (1, 2))
        self.assertEqual(tuple(factor0.values.shape), (2,))
        self.assertEqual(tuple(factor1.values.shape), (2, 2))
        self.assertTrue(
            torch.allclose(
                factor0.values, torch.tensor([0.6, 0.4], dtype=torch.float64)
            )
        )
        self.assertTrue(
            torch.allclose(
                factor1.values,
                torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64),
            )
        )

    def test_read_evidence_file(self):
        with open("examples/simple_model.evid", "r") as f:
            content = f.read().strip()
        self.assertEqual(content, "1 0 1")

        evidence = read_evidence_file("examples/simple_model.evid")
        self.assertEqual(evidence, {1: 1})

    def test_invalid_network_type(self):
        content = "\n".join(
            [
                "INVALID",
                "1",
                "2",
                "0",
            ]
        )
        with self.assertRaises(ValueError):
            read_model_from_string(content)

    def test_scope_size_mismatch(self):
        content = "\n".join(
            [
                "MARKOV",
                "2",
                "2 2",
                "1",
                "2 0",
                "2",
                "0.5 0.5",
            ]
        )
        with self.assertRaises(ValueError):
            read_model_from_string(content)

    def test_missing_table_entries(self):
        content = "\n".join(
            [
                "MARKOV",
                "1",
                "2",
                "1",
                "1 0",
            ]
        )
        with self.assertRaises(ValueError):
            read_model_from_string(content)

    def test_factor_repr(self):
        """Test Factor __repr__ method."""
        values = torch.tensor([0.5, 0.5])
        factor = Factor(vars=[1], values=values)
        repr_str = repr(factor)
        self.assertIn("Factor", repr_str)
        self.assertIn("vars=(1,)", repr_str)
        self.assertIn("shape=", repr_str)

    def test_uai_model_repr(self):
        """Test UAIModel __repr__ method."""
        factor = Factor(vars=[1], values=torch.tensor([0.5, 0.5]))
        model = UAIModel(nvars=1, cards=[2], factors=[factor])
        repr_str = repr(model)
        self.assertIn("UAIModel", repr_str)
        self.assertIn("nvars=1", repr_str)
        self.assertIn("nfactors=1", repr_str)

    def test_read_evidence_empty_filepath(self):
        """Test read_evidence_file with empty filepath."""
        evidence = read_evidence_file("")
        self.assertEqual(evidence, {})

    def test_read_evidence_none_filepath(self):
        """Test read_evidence_file with None filepath."""
        evidence = read_evidence_file(None)
        self.assertEqual(evidence, {})

    def test_read_evidence_empty_file(self):
        """Test read_evidence_file with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.evid', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            evidence = read_evidence_file(temp_path)
            self.assertEqual(evidence, {})
        finally:
            import os
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
