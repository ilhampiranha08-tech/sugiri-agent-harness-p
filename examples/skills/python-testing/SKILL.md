---
name: python-testing
description: Write and run Python tests using pytest. Use when writing tests for Python code.
---

# Python Testing Skill

## Setup

```bash
pip install pytest pytest-cov pytest-asyncio
```

## Writing Tests

### Basic test
```python
def test_addition():
    assert 1 + 1 == 2

def test_string():
    result = "hello".upper()
    assert result == "HELLO"
```

### Fixtures
```python
import pytest

@pytest.fixture
def sample_data():
    return {"name": "test", "value": 42}

def test_with_fixture(sample_data):
    assert sample_data["value"] == 42
```

### Async tests
```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_func()
    assert result == expected
```

### Running tests
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest --cov=src          # With coverage
pytest -k "test_name"     # Run matching tests
pytest tests/test_file.py # Run specific file
```
