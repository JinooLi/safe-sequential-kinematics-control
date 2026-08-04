# Frequently used project commands. Extra options can be passed with ARGS,
# for example: make visualize-clbf ARGS="--show --resolution 160"

UV ?= uv
UV_CACHE_DIR ?= /tmp/safe-inv-clbf-uv-cache
MPLCONFIGDIR ?= /tmp/safe-inv-clbf-matplotlib
export UV_CACHE_DIR MPLCONFIGDIR

.DEFAULT_GOAL := help
.PHONY: help run test train-rdf visualize-rdf visualize-clf visualize-clbf

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

visualize-clf: ## Render a CLF q-space level surface (use ARGS="--show ...")
	$(UV) run python -m src.application.visualize_clf $(ARGS)

visualize-clbf: ## Render x-y, x-yaw, y-yaw CLBF surfaces (use ARGS="--show ...")
	$(UV) run python -m src.application.visualize_clbf $(ARGS)
