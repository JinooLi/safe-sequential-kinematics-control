# Frequently used project commands. Extra options can be passed with ARGS.

UV ?= uv
UV_CACHE_DIR ?= /tmp/safe-inv-clf-uv-cache
MPLCONFIGDIR ?= /tmp/safe-inv-clf-matplotlib
export UV_CACHE_DIR MPLCONFIGDIR

.DEFAULT_GOAL := help
.PHONY: help run test train-rdf visualize-rdf compare-sequential

help: ## Show available command aliases
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

run: ## Run the MuJoCo simulation
	$(UV) run python main.py

test: ## Run all unit tests
	$(UV) run python -m unittest discover -s tests -v

train-rdf: ## Train and validate the Bernstein RDF (use ARGS for options)
	$(UV) run python -m src.application.train_rdf $(ARGS)

visualize-rdf: ## Render the robot distance field (use ARGS="--show ...")
	$(UV) run python -m src.application.visualize_rdf $(ARGS)

compare-sequential-run: ## Run randomized sequential on/off comparison (pass ARGS="--cases 20")
	$(UV) run python -m src.application.compare_sequential run $(ARGS)

compare-sequential-visualize: ## Run visualizer (pass ARGS="--results-dir outputs/<...> --case-id 20")
	$(UV) run python -m src.application.compare_sequential $(ARGS)

compare-sequential-visualize-all: ## Render one comparison image for every completed case
	$(UV) run python -m src.application.compare_sequential visualize-all $(ARGS)
