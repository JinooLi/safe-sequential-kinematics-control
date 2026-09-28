# Robot-specific entry points. Every CLI accepts additional flags via ARGS.
UV ?= uv
UV_CACHE_DIR ?= /tmp/safe-inv-clf-uv-cache
MPLCONFIGDIR ?= /tmp/safe-inv-clf-matplotlib
export UV_CACHE_DIR MPLCONFIGDIR

PANDA_CONFIG ?= configs/panda.yaml
PLANAR_CONFIG ?= configs/planar.yaml
ARGS ?=
RESULTS_DIR ?=
CASE_ID ?=
TRACE ?=
OUTPUT ?=

PANDA = $(UV) run safe-ik --config "$(PANDA_CONFIG)"
PLANAR = $(UV) run python -m safe_ik.application.planar_run --config "$(PLANAR_CONFIG)"
RECORDING = $(UV) run python -m safe_ik.application.recording_view
TRACE_FLAG = $(if $(strip $(TRACE)),--trace "$(TRACE)")
OUTPUT_FLAG = $(if $(strip $(OUTPUT)),--output "$(OUTPUT)")
RESULTS_FLAG = $(if $(strip $(RESULTS_DIR)),--results-dir "$(RESULTS_DIR)")
CASE_FLAG = $(if $(strip $(CASE_ID)),--case-id "$(CASE_ID)")

.DEFAULT_GOAL := help
.PHONY: panda-train-rdf panda-validate-rdf panda-inspect-mesh panda-prepare-mesh
.PHONY: help setup setup-runtime build test check check-format format \
 panda-run panda-view panda-direct panda-direct-view panda-sequential panda-sequential-view panda-plot panda-plot-view \
 planar-run planar-view planar-direct planar-direct-view planar-sequential planar-sequential-view \
 planar-compare planar-compare-plot planar-compare-plot-all planar-train-rdf planar-rdf planar-rdf-view \
 planar-record-plot planar-record-view planar-record-gui planar-record-csv \
 run train-rdf visualize-rdf compare-sequential-run compare-sequential-visualize compare-sequential-visualize-all

help: ## Show grouped commands and optional variables
	@awk 'BEGIN {FS = ":.*##"; print "Usage: make <target> [ARGS=\"CLI options\"]"} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-29s %s\n", $$1, $$2} END {print "\nInputs: PANDA_CONFIG, PLANAR_CONFIG, TRACE, RESULTS_DIR, CASE_ID, OUTPUT"; print "All CLI flags can also be passed in ARGS; ARGS=\"--help\" shows command-specific help."; print "GUI targets require a display. RUN targets use the selected YAML; explicit mode targets override it."}' $(MAKEFILE_LIST)

##@ Panda: simulation and control
panda-inspect-mesh: ## Export standalone HTML highlighting visual-mesh open boundaries
	$(UV) run python -m safe_ik.application.inspect_panda_mesh --config "$(PANDA_CONFIG)" $(OUTPUT_FLAG) $(ARGS)

panda-prepare-mesh: ## Reconstruct visual meshes and export comparison HTML without RDF fitting
	$(UV) run safe-ik-train-panda-rdf --config "$(PANDA_CONFIG)" --prepare-only --output "$(if $(strip $(OUTPUT)),$(OUTPUT),outputs/panda_mesh_reconstruction/reference.npz)" $(ARGS)

panda-train-rdf: ## Offline visual-mesh RDF training and validation (fails closed)
	$(UV) run safe-ik-train-panda-rdf --config "$(PANDA_CONFIG)" $(OUTPUT_FLAG) $(ARGS)

panda-validate-rdf: ## Revalidate existing RDF coefficients without training
	$(UV) run safe-ik-train-panda-rdf --config "$(PANDA_CONFIG)" --validate-only $(OUTPUT_FLAG) $(ARGS)

panda-run: ## Headless run using trajectory.mode from Panda YAML
	$(PANDA) $(ARGS)

panda-view: ## YAML mode with MuJoCo viewer and final diagnostic plot
	$(PANDA) --visualize --show-plot $(ARGS)

panda-direct: ## Headless end-effector pose control, skipping safe IK
	$(PANDA) --mode direct $(ARGS)

panda-direct-view: ## Direct control with viewer and final plot
	$(PANDA) --mode direct --visualize --show-plot $(ARGS)

panda-sequential: ## Headless safe IK followed by sequential CLF-CBF control
	$(PANDA) --mode sequential $(ARGS)

panda-sequential-view: ## Safe IK and sequential control with viewer and final plot
	$(PANDA) --mode sequential --visualize --show-plot $(ARGS)

##@ Panda: saved control traces (no new simulation)
panda-plot: ## Save diagnostic PNG from TRACE=<Panda trajectory.npz>
	$(RECORDING) panda $(TRACE_FLAG) $(OUTPUT_FLAG) $(ARGS)

panda-plot-view: ## Save and show Panda diagnostic plot from TRACE=<...>
	$(RECORDING) panda $(TRACE_FLAG) $(OUTPUT_FLAG) --show $(ARGS)

##@ Planar: single configured trajectory
planar-run: ## Headless run using planar YAML mode and simulation model
	$(PLANAR) $(ARGS)

planar-view: ## YAML mode with MuJoCo viewer and final plot
	$(PLANAR) --visualize --show-plot $(ARGS)

planar-direct: ## Headless direct end-effector control
	$(PLANAR) --mode direct $(ARGS)

planar-direct-view: ## Direct control with viewer and final plot
	$(PLANAR) --mode direct --visualize --show-plot $(ARGS)

planar-sequential: ## Headless sequential link-endpoint control
	$(PLANAR) --mode sequential $(ARGS)

planar-sequential-view: ## Sequential control with viewer and final plot
	$(PLANAR) --mode sequential --visualize --show-plot $(ARGS)

##@ Planar: randomized direct/sequential comparison
planar-compare: ## Run case pairs; ARGS="--cases 20 --workers 4 --duration 10"
	$(UV) run safe-ik-compare run --config "$(PLANAR_CONFIG)" $(ARGS)

planar-compare-plot: ## Render one saved case; RESULTS_DIR=<...> CASE_ID=<n>
	$(UV) run safe-ik-compare visualize $(RESULTS_FLAG) $(CASE_FLAG) $(OUTPUT_FLAG) $(ARGS)

planar-compare-plot-all: ## Render every saved case; RESULTS_DIR=<...>; ARGS="--output-dir ..."
	$(UV) run safe-ik-compare visualize-all $(RESULTS_FLAG) $(ARGS)

##@ Planar: distance-field training and visualization
planar-train-rdf: ## Train/validate Bernstein RDF; OUTPUT overrides the model destination
	$(UV) run safe-ik-train-rdf --config "$(PLANAR_CONFIG)" $(OUTPUT_FLAG) $(ARGS)

planar-rdf: ## Save distance-field PNG; ARGS="--representation capsule|bernstein --q ..."
	$(UV) run safe-ik-visualize-rdf --config "$(PLANAR_CONFIG)" $(OUTPUT_FLAG) $(ARGS)

planar-rdf-view: ## Save and display distance-field plot
	$(UV) run safe-ik-visualize-rdf --config "$(PLANAR_CONFIG)" $(OUTPUT_FLAG) --show $(ARGS)

##@ Planar: saved Recorder data (no new simulation)
planar-record-plot: ## Save plot from TRACE=<planar trajectory.npz>
	$(RECORDING) planar $(TRACE_FLAG) $(OUTPUT_FLAG) $(ARGS)

planar-record-view: ## Save and display plot from TRACE=<...>
	$(RECORDING) planar $(TRACE_FLAG) $(OUTPUT_FLAG) --show $(ARGS)

planar-record-gui: ## Explore recorded signals with buttons and time slider; TRACE=<...>
	$(RECORDING) planar --action gui $(TRACE_FLAG) $(ARGS)

planar-record-csv: ## Export saved trace to CSV; TRACE=<...> [OUTPUT=<...csv>]
	$(RECORDING) planar --action csv $(TRACE_FLAG) $(OUTPUT_FLAG) $(ARGS)

##@ Environment, packaging and development
setup: ## Install locked runtime and development dependencies
	$(UV) sync --locked $(ARGS)

setup-runtime: ## Install locked runtime dependencies without dev tools
	$(UV) sync --locked --no-dev $(ARGS)

build: ## Build source distribution and wheel
	$(UV) build $(ARGS)

test: ## Run the entire regression suite; ARGS="-k parallel" filters tests
	$(UV) run python -m unittest discover -s tests -v $(ARGS)

check-format: ## Check import ordering and Python formatting
	$(UV) run isort --check-only src/safe_ik tests main.py
	$(UV) run black --check src/safe_ik tests main.py

check: check-format test ## Run formatting checks and regression tests

format: ## Format imports and Python source
	$(UV) run isort src/safe_ik tests main.py
	$(UV) run black src/safe_ik tests main.py

##@ Compatibility aliases (prefer the robot-specific names above)
run: panda-run ## Alias of panda-run
train-rdf: planar-train-rdf ## Alias of planar-train-rdf
visualize-rdf: planar-rdf ## Alias of planar-rdf
compare-sequential-run: planar-compare ## Alias of planar-compare
compare-sequential-visualize: ## Legacy dispatcher: ARGS="visualize --results-dir ... --case-id ..."
	$(UV) run safe-ik-compare $(ARGS)
compare-sequential-visualize-all: planar-compare-plot-all ## Alias of planar-compare-plot-all
