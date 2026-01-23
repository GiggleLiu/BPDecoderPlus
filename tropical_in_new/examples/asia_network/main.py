"""Run tropical MPE on a small UAI model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import mpe_tropical, read_model_file  # noqa: E402


def main() -> None:
    model = read_model_file("tropical_in_new/examples/asia_network/model.uai")
    assignment, score, info = mpe_tropical(model)
    print("MPE assignment:", assignment)
    print("MPE log-score:", score)
    print("Info:", info)


if __name__ == "__main__":
    main()
