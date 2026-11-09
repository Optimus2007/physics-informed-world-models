test:
	python -m unittest tests/test_dynamics.py && python -m pytest -q tests
phase_c:
	python -m experiments.phase_c
phase_d:
	python -m experiments.phase_d
phase_d_lowdata:
	python -m experiments.phase_d_lowdata
phase_e:
	python -m experiments.phase_e
phase_f:
	python -m experiments.phase_f
repro:
	python -m experiments.run_all
repro_fast:
	python -m experiments.run_all --fast
repro-fast: repro_fast
report_figs:
	python -m analysis.make_figures
report-figs: report_figs
demo:
	python -m demo.make_gif
report: report_figs
	python -m analysis.make_latex_tables
	cd docs/report_control && tectonic report_control.tex && cp report_control.pdf ../../report_control.pdf
