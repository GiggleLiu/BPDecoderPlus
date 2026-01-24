"""Run tropical MPE on a small UAI model."""

from tropical_in_new.src import mpe_tropical, read_model_file


def main() -> None:
    model = read_model_file("tropical_in_new/examples/asia_network/model.uai")
    assignment, score, info = mpe_tropical(model)
    print("MPE assignment:", assignment)
    print("MPE log-score:", score)
    print("Info:", info)


if __name__ == "__main__":
    main()
