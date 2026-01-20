# Contributing to BPDecoderPlus

Thank you for your interest in contributing to BPDecoderPlus! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/TensorBFS/BPDecoderPlus.git
cd BPDecoderPlus

# Install dependencies with uv (recommended)
uv sync --dev

# Or with pip
pip install -e .[dev,docs]

# Install pre-commit hooks
uv run pre-commit install
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=bpdecoderplus --cov-report=html

# Run specific test file
uv run pytest tests/test_circuit.py

# Run specific test
uv run pytest tests/test_circuit.py::TestCircuitGeneration::test_basic_circuit
```

## Code Style

This project uses:

- **Ruff** for linting and formatting (replaces black, flake8, isort)
- **Type hints** for public APIs
- **pytest** for testing
- **Google-style docstrings**

### Before Submitting

```bash
# Format code with ruff
uv run ruff format .

# Check linting
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .

# Run pre-commit hooks on all files
uv run pre-commit run --all-files
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Write tests** for your changes
   - Add tests in the `tests/` directory
   - Maintain or improve code coverage

3. **Update documentation** if needed
   - Update docstrings for modified functions
   - Update `docs/` if adding new features
   - Update README.md if changing user-facing behavior

4. **Run the test suite** and ensure all tests pass
   ```bash
   uv run pytest
   ```

5. **Run pre-commit hooks** to ensure code quality
   ```bash
   uv run pre-commit run --all-files
   ```

6. **Submit a Pull Request** with:
   - Clear description of changes
   - Link to related issues (if any)
   - Screenshots/examples for UI changes

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): brief description

Longer description if needed, explaining what changed and why.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring (no functional changes)
- `perf`: Performance improvements
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

**Examples:**
```
feat(circuit): add support for code distance 9

fix(dem): correct parity check matrix generation for edge cases

docs(readme): update installation instructions for uv

test(syndrome): add tests for large syndrome databases
```

## Development Guidelines

### Adding New Features

1. **Discuss first** for major features
   - Open an issue to discuss the design
   - Get feedback before implementing

2. **Write tests** before implementation (TDD encouraged)
   - Unit tests for individual functions
   - Integration tests for workflows

3. **Update documentation**
   - Add docstrings with examples
   - Update `docs/` guides
   - Add usage examples if helpful

4. **Keep it simple**
   - Follow existing patterns in the codebase
   - Don't over-engineer solutions
   - Prefer clarity over cleverness

### Project Structure

```
BPDecoderPlus/
├── src/bpdecoderplus/          # Main package code
│   ├── circuit.py              # Circuit generation with Stim
│   ├── dem.py                  # Detector error model export
│   ├── syndrome.py             # Syndrome sampling
│   ├── cli.py                  # Command-line interface
│   └── pytorch_bp/             # Belief propagation module
│       ├── belief_propagation.py
│       └── uai_parser.py
├── tests/                      # Test suite
│   ├── test_circuit.py
│   ├── test_dem.py
│   ├── test_syndrome.py
│   ├── test_bp_basic.py
│   └── test_integration.py
├── docs/                       # MkDocs documentation
│   ├── getting_started.md
│   ├── usage_guide.md
│   └── api_reference.md
├── examples/                   # Usage examples
│   ├── simple_example.py
│   └── evidence_example.py
└── datasets/                   # Sample datasets
```

### Testing Guidelines

- **Write unit tests** for new functions
  - Test happy path and edge cases
  - Test error conditions

- **Use pytest fixtures** for common setups
  - Add fixtures to `tests/conftest.py`

- **Aim for >80% code coverage**
  - Check coverage with `pytest --cov`

- **Test documentation examples**
  - Ensure code examples in docs actually work

### Documentation Guidelines

- **Use Google-style docstrings**
  ```python
  def generate_circuit(distance: int, rounds: int) -> stim.Circuit:
      """Generate a noisy surface code circuit.

      Args:
          distance: Code distance (odd integer >= 3)
          rounds: Number of measurement rounds

      Returns:
          Stim circuit with noise model applied

      Raises:
          ValueError: If distance is invalid

      Example:
          >>> circuit = generate_circuit(distance=3, rounds=5)
          >>> print(circuit.num_qubits)
          17
      """
  ```

- **Include type hints** in function signatures
- **Add usage examples** for public APIs
- **Update `docs/`** for major features

## Release Process

Releases are managed by project maintainers:

1. Update `CHANGELOG.md` with changes
2. Bump version in `src/bpdecoderplus/__init__.py`
3. Create git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions automatically publishes to PyPI

## Reporting Issues

### Bug Reports

Please include:

- **Description** of the bug
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment** (OS, Python version)
- **Code sample** or error message

### Feature Requests

Please include:

- **Use case** - what problem does this solve?
- **Proposed solution** - how should it work?
- **Alternatives** - other approaches considered

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/TensorBFS/BPDecoderPlus/issues)
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: [Online Docs](https://tensorbfs.github.io/BPDecoderPlus/)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
Please read and follow it in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to BPDecoderPlus! 🎉
