# Contributing to PonderTTT

Thank you for your interest in contributing to PonderTTT!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ponderttt.git
cd ponderttt
```

2. Install dependencies:
```bash
make install
```

Or manually:
```bash
uv pip install -e .
uv pip install -e ".[dev]"
```

3. Run tests:
```bash
make test
```

## Code Style

We use the following tools to maintain code quality:

- **Ruff**: Linting and formatting
- **MyPy**: Static type checking
- **Pytest**: Testing

### Running Linters

```bash
# Format code
make format

# Run linters
make lint
```

### Code Formatting Guidelines

- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Follow PEP 8 style guidelines
- Maximum line length: 100 characters (enforced by ruff)

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest tests/ --cov=src/ponderttt
```

### Writing Tests

- Place test files in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use fixtures for common setup
- Aim for >80% code coverage

Example:
```python
def test_model_forward_pass(base_model, tokenizer):
    """Test that model can perform forward pass."""
    text = "def test(): pass"
    encoded = tokenizer(text, return_tensors="pt")
    outputs = base_model(**encoded)
    assert outputs["logits"] is not None
```

## Project Structure

```
ponderttt/
├── src/ponderttt/       # Main package
│   ├── data/            # Data loading
│   ├── models/          # Model architectures
│   ├── training/        # Training algorithms
│   ├── evaluation/      # Evaluation metrics
│   └── utils/           # Utilities
├── tests/               # Unit tests
├── scripts/             # Helper scripts
└── docs/                # Documentation
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests and linters: `make test && make lint`
5. Commit your changes: `git commit -am 'Add new feature'`
6. Push to the branch: `git push origin feature/your-feature`
7. Create a Pull Request

### PR Guidelines

- Write clear, descriptive commit messages
- Include tests for new features
- Update documentation as needed
- Ensure all tests pass
- Keep PRs focused and atomic

## Bug Reports

When filing a bug report, please include:

1. Description of the bug
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, etc.)

## Feature Requests

We welcome feature requests! Please:

1. Check if the feature already exists or is planned
2. Describe the feature and its use case
3. Explain how it fits with the project goals
4. Provide examples if applicable

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose. Reviewers will check for:

- Code quality and style
- Test coverage
- Documentation
- Compatibility with existing code
- Performance implications

## Questions?

Feel free to open an issue for questions or discussion!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
