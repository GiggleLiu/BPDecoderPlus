# BPDecoderPlus Makefile
# Winter School Project: BP Decoder for Surface Codes
#
# Usage:
#   make setup      - Install all dependencies
#   make test       - Run unit tests
#   make benchmark  - Generate benchmark data
#   make visualize  - Create plots from benchmark data
#   make all        - Run full pipeline (test -> benchmark -> visualize)
#   make quick      - Run quick version (fewer trials)
#   make clean      - Remove generated files

.PHONY: all setup test benchmark visualize quick clean help circuit-data

# Default target
all: test benchmark visualize

# Help message
help:
	@echo "BPDecoderPlus - Winter School Project"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup           - Install Julia and Python dependencies"
	@echo "  make test            - Run unit tests"
	@echo "  make benchmark       - Generate benchmark data (full)"
	@echo "  make quick           - Run quick benchmark (fewer trials)"
	@echo "  make visualize       - Create plots from benchmark data"
	@echo "  make all             - Run full pipeline"
	@echo "  make circuit-data    - Generate DEM + syndrome datasets (Stim)"
	@echo "  make clean           - Remove generated files"
	@echo ""
	@echo "Circuit-level data generation:"
	@echo "  make circuit-data           - Full dataset (d=3,5,7,9, 100k shots)"
	@echo "  make circuit-data-quick     - Quick dataset (d=3,5, 10k shots)"
	@echo "  make circuit-data-custom D=5 R=5 P=0.001 N=100000"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup && make quick && make visualize"

# Setup: Install dependencies
setup: setup-julia setup-python

setup-julia:
	@echo ">>> Installing Julia dependencies..."
	julia --project=. -e 'using Pkg; Pkg.instantiate()'
	@echo ">>> Julia setup complete"

setup-python:
	@echo ">>> Installing Python dependencies..."
	pip install -r python/requirements.txt
	@echo ">>> Python setup complete"

# Run tests
test:
	@echo ">>> Running tests..."
	julia --project=. test/runtests.jl

# Generate benchmark data (full)
benchmark: benchmark-data benchmark-timing

benchmark-data:
	@echo ">>> Generating benchmark data..."
	julia --project=. benchmark/generate_data.jl

benchmark-timing:
	@echo ">>> Running timing benchmarks..."
	julia --project=. benchmark/run_benchmarks.jl

# Quick mode (fewer trials for faster iteration)
quick: quick-benchmark quick-visualize

quick-benchmark:
	@echo ">>> Running quick benchmarks..."
	julia --project=. benchmark/generate_data.jl --quick
	julia --project=. benchmark/run_benchmarks.jl --quick

quick-visualize: visualize

# Visualization
visualize:
	@echo ">>> Generating plots..."
	@mkdir -p results/plots
	python python/visualize.py --data-dir benchmark/data --output-dir results/plots
	@echo ">>> Plots saved to results/plots/"

# Clean generated files
clean:
	@echo ">>> Cleaning generated files..."
	rm -rf benchmark/data/*.json
	rm -rf benchmark/circuit_data/
	rm -rf results/plots/*.png
	rm -rf Manifest.toml
	@echo ">>> Clean complete"

# Development helpers
dev-repl:
	@echo ">>> Starting Julia REPL..."
	julia --project=.

dev-precompile:
	@echo ">>> Precompiling package..."
	julia --project=. -e 'using Pkg; Pkg.precompile()'

# Run a single decoder benchmark
benchmark-ip:
	@echo ">>> Running IP decoder benchmark..."
	julia --project=. -e 'using BPDecoderPlus; println(quick_benchmark(decoder=:IP))'

benchmark-bp:
	@echo ">>> Running BP decoder benchmark..."
	julia --project=. -e 'using BPDecoderPlus; println(quick_benchmark(decoder=:BP))'

benchmark-bposd:
	@echo ">>> Running BP+OSD decoder benchmark..."
	julia --project=. -e 'using BPDecoderPlus; println(quick_benchmark(decoder=:BPOSD))'

# Compare decoders
compare:
	@echo ">>> Comparing decoders..."
	julia --project=. -e 'using BPDecoderPlus; compare_decoders([3,5], [0.02,0.05,0.08], 100)'

# Circuit-level data generation (DEM + syndrome datasets)
circuit-data:
	@echo ">>> Generating circuit-level datasets (DEM + syndromes)..."
	@mkdir -p benchmark/circuit_data
	python python/generate_circuit_data.py
	@echo ">>> Circuit data saved to benchmark/circuit_data/"

circuit-data-quick:
	@echo ">>> Generating circuit-level datasets (quick mode)..."
	@mkdir -p benchmark/circuit_data
	python python/generate_circuit_data.py --quick
	@echo ">>> Circuit data saved to benchmark/circuit_data/"

circuit-data-custom:
	@echo ">>> Generating custom circuit-level dataset..."
	@echo "    Usage: make circuit-data-custom D=5 R=5 P=0.001 N=100000"
	@mkdir -p benchmark/circuit_data
	python python/generate_circuit_data.py --distance $(D) --rounds $(R) --p-error $(P) --shots $(N)
	@echo ">>> Circuit data saved to benchmark/circuit_data/"

# Atom loss datasets using TensorQEC
atomloss-data:
	@echo ">>> Generating atom loss syndrome datasets (TensorQEC)..."
	@mkdir -p benchmark/circuit_data
	julia --project=. benchmark/generate_atom_loss_syndromes.jl
	@echo ">>> Atom loss data saved to benchmark/circuit_data/"

atomloss-data-quick:
	@echo ">>> Generating atom loss syndrome datasets (quick mode)..."
	@mkdir -p benchmark/circuit_data
	julia --project=. benchmark/generate_atom_loss_syndromes.jl --quick
	@echo ">>> Atom loss data saved to benchmark/circuit_data/"

# Circuit visualization using Stim
visualize-circuit:
	@echo ">>> Generating circuit visualization with Stim..."
	@mkdir -p results/plots
	python python/visualize.py circuit --distance 3 --rounds 1 --diagram all --samples 100
	@echo ">>> Circuit visualizations saved to results/plots/"

visualize-circuit-large:
	@echo ">>> Generating larger circuit visualization..."
	@mkdir -p results/plots
	python python/visualize.py circuit --distance 5 --rounds 3 --diagram all --samples 1000
	@echo ">>> Circuit visualizations saved to results/plots/"
