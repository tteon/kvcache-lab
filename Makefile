
# vLLM LMCache Experiment Lab Makefile

SHELL := /bin/bash

# Default targets
.PHONY: all help baseline cpu disk scalability analyze clean

help:
	@echo "vLLM Experiment Lab"
	@echo "Available commands:"
	@echo "  make baseline      - Run Baseline (GPU) benchmark"
	@echo "  make cpu          - Run CPU Offload benchmark"
	@echo "  make disk         - Run Disk Offload benchmark"
	@echo "  make analyze      - Generate comparison plots and report"
	@echo "  make calculator   - Run interactive Bottleneck Calculator"
	@echo "  make clean        - Remove cache and temporary results"

baseline:
	./run_experiments.sh --tier baseline

cpu:
	./run_experiments.sh --tier cpu

disk:
	./run_experiments.sh --tier disk

scalability:
	./run_experiments.sh --tier scalability

analyze:
	python3 analysis/plot_results.py --input "results/*/metrics_*.csv" --output-prefix "final_comparison"
	@echo "Analysis saved to comparison_*.png and comparison_report.md"

calculator:
	python3 analysis/bottleneck_calculator.py --mode both --N 2000 --R 50 --T 100 --alpha 0.8

clean:
	rm -rf cache_store/*
	@echo "Cache cleared."
