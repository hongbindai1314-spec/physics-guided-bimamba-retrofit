.PHONY: install test validate clean

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	pytest -q

validate:
	python scripts/run_validation_analysis.py

clean:
	rm -rf .pytest_cache __pycache__ src/gbc_bep/__pycache__ tests/__pycache__

.PHONY: benchmark benchmark-test benchmark-verify
benchmark:
	python scripts/run_simulation.py
benchmark-test:
	python scripts/run_simulation.py --mode tests
benchmark-verify:
	python scripts/run_simulation.py --mode verify
