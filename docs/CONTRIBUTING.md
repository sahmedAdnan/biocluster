# Contributing

```bash
git clone https://github.com/sahmedAdnan/biocluster.git
cd biocluster
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[gui,dev]"
pytest -q
ruff check src tests scripts examples
```

## Ground rules

- **Tests must not download model weights or read a dataset.** The suite runs on
  synthetic embeddings; keep it that way so CI stays fast and contributors
  without the checkpoints can still verify their work.
- **One interface per concern.** New clustering algorithms go in
  `clustering/methods.py` behind `run_clustering`; new reduction modes in
  `reduction/reduction.py` behind `reduce_features`; new encoders in
  `models/extractors.py` behind `build_extractor`. The GUI and the CLI both call
  those entry points, so anything added behind them works in both.
- **Do not change metric conventions silently.** Accuracy excludes noise points;
  tier counts include them. If you alter either, update `docs/PIPELINE.md` in the
  same commit, because published numbers depend on these definitions.

## Known cleanup debt

- `src/biocluster/app/main.py` is a ~3,300-line Qt application and is excluded
  from linting. It is covered by an import test, not by style rules. Splitting
  the worker threads, the tabs and the review gallery into separate modules is
  the obvious next refactor.
- `app/theme.py` injects its palette with `globals().update(...)`, which no static
  checker can follow. An explicit dataclass palette would remove the need for the
  `F821` exemption.
