# Contributing

Thank you for considering contributing to TDA-Longevity-Resilience!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/tda-longevity-resilience.git`
3. Create a branch: `git checkout -b feature/my-feature`
4. Set up pre-commit hooks: `pre-commit install`
5. Make your changes
6. Run tests: `pytest tests/ -v`
7. Commit: `git commit -m "feat: add my feature"`
8. Push: `git push origin feature/my-feature`
9. Open a Pull Request

## Code Style

- Python 3.10+ with type hints where helpful
- [PEP 8](https://peps.python.org/pep-0008/) with max line length 120
- Format with `black --line-length=120`
- Sort imports with `isort --profile=black`
- Lint with `flake8 --max-line-length=120 --extend-ignore=E203,W503`
- Use docstrings (NumPy style) for all public functions

## Testing

- Write tests for new features in `tests/`
- Use `pytest` with descriptive test names
- Aim for >80% coverage
- Run `pytest --cov=src` to check coverage

## Pull Request Process

1. Ensure all tests pass and coverage is maintained
2. Update relevant documentation if needed
3. Add a CHANGELOG entry
4. Request review from a maintainer
5. Address review comments

## Reporting Bugs

Open an issue using the Bug Report template. Include:
- Steps to reproduce
- Expected behavior
- Environment details (OS, Python version, package versions)
- Any relevant persistence diagram `.pkl` files

## Feature Requests

Open an issue using the Feature Request template. Describe:
- The problem you're solving
- Your proposed solution
- Alternatives you've considered
- TDA-specific context (omics layer, homology dimension, etc.)

## Questions?

Open a [Discussion](https://github.com/jbrandonp/tda-longevity-resilience/discussions) on GitHub.
