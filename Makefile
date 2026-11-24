.PHONY: install test verify bench scale report demo

install:
	pip install -c constraints.txt -e ".[dev]"

# Recomputes every summary from results/runs.json and reruns a fixed subset of tasks.
test:
	pytest -q

# Reruns every task and checks it against results/runs.json (several minutes).
verify:
	FULL=1 pytest -q tests/test_results.py

# Offline, deterministic: development choices, then held-out seeds once.
# Writes results/{runs,dev_choices,sweep,base}.json and RESULTS.md.
bench:
	python -m clusterfi.cli bench --out results

# Machine-specific: parallel permutation importance versus worker count.
scale:
	python -m clusterfi.cli scale --out results --workers 1,2,4,8,16

report:
	python -m clusterfi.cli report --out results

demo:
	python -m clusterfi.cli demo --seed 1
