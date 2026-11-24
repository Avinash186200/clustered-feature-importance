# Contributing

Thanks for taking a look. Changes are judged on whether they keep the repo offline, honest about its numbers, and free of test leakage.

## Ground rules

1. **Everything in CI runs offline and on CPU.** Tests and `make bench` need no network and no GPU.
2. **Numbers come from results.** Any number in a doc must be a field in a `results/*.json` file written by `make bench` or `make scale`. If you change code that affects results, rerun `make bench`, commit the new JSON and `RESULTS.md` with the change, and run `make verify`.
3. **Never tune on held-out seeds.** The clustering rule, model settings, thresholds and any new design choice are made on the development seeds (1-5). If a change needs a new choice after held-out results exist, make it on development seeds, draw fresh held-out seeds, and record the history in `docs/methodology.md`.
4. **No timing assertions in tests.** CI runners are shared and slow; tests check correctness and equivalence, never speed. `make scale` numbers are printed and committed, never asserted.
5. **Determinism.** Every generator and permutation takes a seed; LightGBM runs single-threaded with `deterministic=True`. Parallel and serial runs must give identical numbers.
6. **No secrets.** Nothing here needs credentials; `.env` is git-ignored and CI runs a secret scan.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[dev]"
pytest -q
```

## Adding an importance method

1. Add a function to `importance.py` that returns one score per unit (feature or cluster).
2. Register it in `pipeline.run_methods` (and in `PLAIN` or `CLUSTERED`).
3. Add a unit test in `tests/test_importance.py`, rerun `make bench`, and update the docs from the new JSON.
