# mlops-lab

Learning lab: take a scikit-learn model and wrap it in the practices of a production ML system
(tracking, versioning, serving, CI/CD, monitoring). Built week by week.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install     # deps + pre-commit hooks
make check       # ruff + pytest
make train       # writes artifacts/model.joblib and artifacts/metrics.json
```

## Layout

```
src/mlops_lab/data.py    load and split data (swap in your course dataset here)
src/mlops_lab/train.py   build pipeline, train, evaluate, save artifacts
tests/                   unit tests, including a quality floor on the trained model
.github/workflows/ci.yml lint -> test -> train, uploads the model as a build artifact
```

## Roadmap

| Week | Topic | Status |
|---|---|---|
| 1 | Repo, tests, lint, CI | done |
| 2 | Config and project structure | |
| 3-4 | MLflow tracking and registry | |
| 5 | DVC data and pipeline | |
| 6-8 | Serving: FastAPI, Docker, Helm, kind | |
| 9+ | GKE, CI/CD with model gate, orchestration, monitoring, GPU | |
