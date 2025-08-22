.PHONY: fpl-bootstrap fpl-gw fpl-player-hist fpl-refresh fpl-features fpl-merge fpl-validate fpl-featurize fpl-optimize fpl-rotate fpl-sim fpl-eval fpl-benchmark test

ENV ?= .env

fpl-bootstrap:
	python -m scripts.fpl.fpl_bootstrap

fpl-gw:
	@if [ -z "$$GW" ]; then echo "Usage: make fpl-gw GW=1"; exit 1; fi
	python -m scripts.fpl.fetch_gw --gw $$GW

fpl-player-hist:
	python -m scripts.fpl.fetch_player_summaries

fpl-features:
	@echo "Building FPL features (Step 2b)..."
	python -m src.features.fpl_features
	@echo "FPL features built successfully!"

fpl-merge:
	@echo "Running FPL Entity Resolution & Merge (Step 3a)..."
	python -m scripts.fpl.merge_fpl
	@echo "FPL merge completed successfully!"

fpl-validate:
	@echo "Running FPL Merge Validation (Step 3b)..."
	python -m scripts.fpl.validate_merge
	@echo "Validation completed!"

fpl-featurize:
	@echo "Building model-ready features (Step 3c)..."
	python -m scripts.fpl.build_features
	@echo "Features ready at data/FPL/processed/fpl_features_model.parquet"

fpl-optimize:
	@echo "Running FPL Squad Optimization (Step 4a)..."
	python -m scripts.fpl.optimizer
	@echo "FPL squad optimization completed!"

fpl-rotate:
	@echo "Running FPL Squad Rotation (Step 4b)..."
	python -m scripts.fpl.rotation
	@echo "FPL squad rotation completed!"

fpl-sim:
	@if [ -z "$$GW_START" ] || [ -z "$$GW_END" ]; then echo "Usage: make fpl-sim GW_START=1 GW_END=6"; exit 1; fi
	python -m scripts.fpl.simulate --gw-start $${GW_START} --gw-end $${GW_END}

fpl-eval:
	@echo "Running FPL Evaluation & Stress Tests (Step 5a)..."
	python -m scripts.fpl.evaluate
	@echo "FPL evaluation completed!"

fpl-benchmark:
	@if [ -z "$$GW_START" ] || [ -z "$$GW_END" ]; then echo "Usage: make fpl-benchmark GW_START=1 GW_END=10"; exit 1; fi
	@echo "Running FPL Benchmarks vs Historical Seasons (Step 5b)..."
	python -m scripts.fpl.benchmark --gw-start $${GW_START} --gw-end $${GW_END}
	@echo "FPL benchmarking completed!"

fpl-refresh:
	@if [ -z "$$GW" ]; then echo "Usage: make fpl-refresh GW=1"; exit 1; fi
	@echo "Running complete FPL data refresh for GW$$GW..."
	@make fpl-bootstrap
	@make fpl-gw GW=$$GW
	@make fpl-player-hist
	@echo "FPL refresh completed successfully!"

test:
	pytest -q 