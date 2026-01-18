import unittest
import torch

from pytorch_bp import read_model_from_string, read_evidence_file


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


if __name__ == "__main__":
    unittest.main()
