.PHONY: install lint format test train check

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

train:
	python -m mlops_lab.train

check: lint test
