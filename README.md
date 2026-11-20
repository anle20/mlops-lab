# mlops-lab

Learning lab: take a scikit-learn model and wrap it in the practices of a production ML system
(tracking, versioning, serving, CI/CD, monitoring). Built week by week.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install     # deps + pre-commit hooks
make check       # ruff + pytest
make train       # writes artifacts/model.joblib and artifacts/metrics.json
make predict     # predicts on examples/sample.csv with the trained model
```

## Changing parameters

Edit `configs/train.yaml` (for example `model.c`) and run `make train`. The file is validated by
pydantic, so a typo in a key or an out-of-range value fails with a clear error.

## Layout

```
configs/train.yaml        training parameters
src/mlops_lab/config.py   pydantic models + YAML loader
src/mlops_lab/data.py     load and split data (swap in your course dataset here)
src/mlops_lab/train.py    build pipeline, train, evaluate, save artifacts
src/mlops_lab/predict.py  load model, validate input columns, predict (reused by the API in week 6)
examples/sample.csv       a few rows used by `make predict` and CI
tests/                    unit tests, including a quality floor on the trained model
.github/workflows/ci.yml  lint -> test -> train -> predict, uploads the model as a build artifact
```

## Roadmap

| Week | Topic | Status |
|---|---|---|
| 1 | Repo, tests, lint, CI | done |
| 2 | Config and project structure | done |
| 3-4 | MLflow tracking and registry | |
| 5 | DVC data and pipeline | |
| 6-8 | Serving: FastAPI, Docker, Helm, kind | |
| 9+ | GKE, CI/CD with model gate, orchestration, monitoring, GPU | |
