test:
	python -m unittest tests/test_dynamics.py && python -m pytest -q tests
repro-fast:
	python experiments/run_all.py --fast
repro:
	python experiments/run_all.py
report-figs:
	python analysis/make_figures.py
