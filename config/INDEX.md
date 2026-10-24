# `config/` — experiment run configurations

Plain YAML files for experiment stacks (`run.sh`). Copy from `experiments-template/` into `experiments/`, tweak, and pass the path to the experiment runner.

| Path | Role |
|---|---|
| [`experiments-template/chess.yaml`](experiments-template/chess.yaml) | Documented template — copy before customizing |
| [`experiments/chess.yaml`](experiments/chess.yaml) | Default chess run (qwen3 profile) |

Main application settings remain in repo-root [`config.yaml`](../config.yaml). Start an experiment with [`run.sh`](../run.sh):

```bash
./run.sh config/experiments/chess.yaml start
```

Each experiment YAML must set `experiment`, `entrypoint` (Python setup module), and an experiment-named settings block (e.g. `chess:`).
