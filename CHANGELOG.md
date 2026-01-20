# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Reorganized dataset structure to single-source flat layout in `datasets/` directory
- Updated CLI default dataset path from `datasets/noisy_circuits` to `datasets`
- Moved all visualization images to `datasets/visualizations/` subdirectory
- Updated package metadata in `pyproject.toml` with project URLs and maintainers
- Updated README.md to focus on Python implementation
- Updated dataset documentation to reflect new structure

### Added
- LICENSE file (MIT License)
- CONTRIBUTING.md with development guidelines
- Project URLs in package metadata (Homepage, Documentation, Repository, Issues, Changelog)
- Maintainers field in package metadata

### Removed
- Duplicate dataset files from subdirectories (`circuits/`, `dems/`, `uais/`, `syndromes/`, `noisy_circuits/`)
- Julia code examples and references from README
- Winter school training references from project description

## [0.1.0] - 2025-01-20

### Added
- Initial release of BPDecoderPlus
- Noisy circuit generation for surface codes using Stim
- Belief propagation decoder implementation using PyTorch
- CLI tool for generating noisy circuits and detector error models
- Support for surface code distances 3, 5, 7, 9
- Syndrome database generation and storage
- UAI format export for inference problems
- PyTorch-based BP solver with customizable iterations
- GitHub Pages documentation with MkDocs
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions
- Example datasets for d=3 surface codes with varying rounds

### Features
- Circuit-level surface code simulation
- Detector error model (DEM) generation
- Syndrome extraction and database management
- Multiple output formats: Stim circuits, DEM files, UAI files, NPZ syndrome databases
- Configurable physical error rates and code parameters
- Integration with Stim for fast quantum circuit simulation
