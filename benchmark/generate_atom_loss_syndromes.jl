#!/usr/bin/env julia
"""
Generate syndrome datasets with atom loss using TensorQEC.

This script generates circuit-level syndrome data with atom loss errors,
compatible with the Stim-based datasets.

Usage:
    julia benchmark/generate_atom_loss_syndromes.jl [--quick]
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using TensorQEC
using Random
using JSON
using Printf
using Statistics

const QUICK_MODE = "--quick" in ARGS
const OUTPUT_DIR = joinpath(dirname(@__FILE__), "circuit_data")
const SEED = 42

"""
Configuration for atom loss dataset generation.
"""
struct AtomLossConfig
    distance::Int
    rounds::Int
    p_error::Float64      # Base depolarizing error rate
    p_meas::Float64       # Measurement error rate
    p_loss_1q::Float64    # Single-qubit gate atom loss probability
    p_loss_2q::Float64    # Two-qubit gate atom loss probability
    num_shots::Int
end

"""
Generate a surface code syndrome extraction circuit with detectors.
"""
function generate_surface_code_circuit(d::Int, rounds::Int;
                                       p_error::Float64=0.001,
                                       p_meas::Float64=0.001,
                                       p_reset::Float64=0.001)
    # Create surface code
    code = SurfaceCode(d, d)
    tanner = CSSTannerGraph(code)

    # For now, we'll generate syndromes directly using TensorQEC's code-capacity model
    # and simulate atom loss effects
    return code, tanner
end

"""
Sample syndromes with atom loss effects.

Atom loss is modeled as:
1. With probability p_loss, a qubit is lost
2. Lost qubits contribute random errors (erasure -> depolarizing)
3. The decoder may or may not know which qubits are lost
"""
function sample_with_atom_loss(tanner::CSSTannerGraph, config::AtomLossConfig;
                               seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    d = config.distance
    n_qubits = d^2
    n_rounds = config.rounds

    # Number of detectors = stabilizers × rounds (approximately)
    n_X = size(tanner.stgx.H, 1)
    n_Z = size(tanner.stgz.H, 1)
    n_stabilizers = n_X + n_Z
    n_detectors = n_stabilizers * n_rounds

    # Storage for results
    detection_events = zeros(Int, config.num_shots, n_detectors)
    observable_flips = zeros(Int, config.num_shots)
    loss_masks = zeros(Int, config.num_shots, n_qubits)  # Which qubits were lost

    # Get logical operators
    lx, lz = logical_operator(tanner)

    for shot in 1:config.num_shots
        # Track cumulative error on data qubits
        # Pauli frame: 0=I, 1=X, 2=Z, 3=Y
        current_error = zeros(Int, n_qubits)

        # Track which qubits are lost
        lost_qubits = Set{Int}()

        # Previous syndrome for detection events
        prev_syndrome = zeros(Int, n_stabilizers)

        detector_idx = 1

        for round in 1:n_rounds
            # 1. Apply atom loss (qubits can be lost during gates)
            # Simplified model: each qubit has probability p_loss of being lost per round
            p_loss_per_round = 1 - (1 - config.p_loss_2q)^4  # ~4 two-qubit gates per round
            for q in 1:n_qubits
                if !(q in lost_qubits) && rand() < p_loss_per_round
                    push!(lost_qubits, q)
                    # Lost qubit contributes random Pauli error
                    current_error[q] = rand(0:3)
                end
            end

            # 2. Apply depolarizing errors on non-lost qubits
            for q in 1:n_qubits
                if !(q in lost_qubits) && rand() < config.p_error
                    pauli = rand(1:3)  # X, Z, or Y
                    current_error[q] = xor(current_error[q], pauli)
                end
            end

            # 3. Compute syndrome
            error_X = [(e & 1) != 0 for e in current_error]
            error_Z = [(e & 2) != 0 for e in current_error]

            H_X = Int.(Matrix(tanner.stgx.H))
            H_Z = Int.(Matrix(tanner.stgz.H))

            syndrome_X = mod.(H_X * Int.(error_Z), 2)
            syndrome_Z = mod.(H_Z * Int.(error_X), 2)
            current_syndrome = vcat(syndrome_X, syndrome_Z)

            # 4. Apply measurement errors
            for s in 1:n_stabilizers
                if rand() < config.p_meas
                    current_syndrome[s] = 1 - current_syndrome[s]
                end
            end

            # 5. Compute detection events (syndrome differences)
            for s in 1:n_stabilizers
                detection_events[shot, detector_idx] = mod(current_syndrome[s] - prev_syndrome[s], 2)
                detector_idx += 1
            end

            prev_syndrome = copy(current_syndrome)
        end

        # 6. Check logical error
        # Construct error pattern for TensorQEC using Mod2 vectors
        xerror = Mod2.([(current_error[i] & 1) != 0 for i in 1:n_qubits])
        zerror = Mod2.([(current_error[i] & 2) != 0 for i in 1:n_qubits])

        # Zero correction
        zero_xerror = zeros(Mod2, n_qubits)
        zero_zerror = zeros(Mod2, n_qubits)

        # Check logical error: X errors checked against lz, Z errors against lx
        has_x_logical = any(i -> sum(lz[i, :] .* xerror).x, 1:size(lz, 1))
        has_z_logical = any(i -> sum(lx[i, :] .* zerror).x, 1:size(lx, 1))

        observable_flips[shot] = (has_x_logical || has_z_logical) ? 1 : 0

        # Record loss mask
        for q in lost_qubits
            loss_masks[shot, q] = 1
        end
    end

    return detection_events, observable_flips, loss_masks
end

"""
Save dataset in Stim-compatible format.
"""
function save_dataset(output_dir::String, prefix::String, config::AtomLossConfig,
                      detection_events::Matrix{Int}, observable_flips::Vector{Int},
                      loss_masks::Matrix{Int})
    mkpath(output_dir)

    num_shots, num_detectors = size(detection_events)
    num_qubits = size(loss_masks, 2)
    logical_error_rate = mean(observable_flips)
    avg_loss_rate = mean(loss_masks)

    # Save detection events (.01 format)
    events_path = joinpath(output_dir, "$(prefix)_events.01")
    open(events_path, "w") do f
        for shot in 1:num_shots
            println(f, join(detection_events[shot, :], ""))
        end
    end

    # Save observable flips
    obs_path = joinpath(output_dir, "$(prefix)_obs.01")
    open(obs_path, "w") do f
        for shot in 1:num_shots
            println(f, observable_flips[shot])
        end
    end

    # Save loss masks (which qubits were lost)
    loss_path = joinpath(output_dir, "$(prefix)_loss.01")
    open(loss_path, "w") do f
        for shot in 1:num_shots
            println(f, join(loss_masks[shot, :], ""))
        end
    end

    # Save metadata
    metadata = Dict(
        "code" => "surface_code:rotated_memory_z",
        "distance" => config.distance,
        "rounds" => config.rounds,
        "p_error" => config.p_error,
        "p_meas" => config.p_meas,
        "p_loss_1q" => config.p_loss_1q,
        "p_loss_2q" => config.p_loss_2q,
        "num_shots" => num_shots,
        "num_detectors" => num_detectors,
        "num_qubits" => num_qubits,
        "num_observables" => 1,
        "logical_error_rate" => logical_error_rate,
        "avg_loss_rate" => avg_loss_rate,
        "generator" => "TensorQEC (atom loss)",
        "files" => Dict(
            "events_01" => "$(prefix)_events.01",
            "observables" => "$(prefix)_obs.01",
            "loss_mask" => "$(prefix)_loss.01"
        )
    )

    metadata_path = joinpath(output_dir, "$(prefix)_metadata.json")
    open(metadata_path, "w") do f
        JSON.print(f, metadata, 2)
    end

    return metadata
end

function main()
    Random.seed!(SEED)

    println("=" ^ 60)
    println("Atom Loss Syndrome Dataset Generation (TensorQEC)")
    println("=" ^ 60)

    # Configuration
    if QUICK_MODE
        distances = [3, 5]
        rounds_list = [3, 5]
        p_errors = [0.001, 0.005]
        p_loss_rates = [0.0, 0.01, 0.02]
        num_shots = 10000
        println("Running in QUICK mode")
    else
        distances = [3, 5, 7]
        rounds_list = [3, 5, 7]
        p_errors = [0.001, 0.005, 0.01]
        p_loss_rates = [0.0, 0.005, 0.01, 0.02, 0.03]
        num_shots = 50000
    end

    p_meas = 0.001  # Fixed measurement error rate

    println("Output directory: $OUTPUT_DIR")
    println("Distances: $distances")
    println("Loss rates: $p_loss_rates")
    println("Shots per config: $num_shots")
    println("=" ^ 60)

    all_metadata = []

    for (d, r) in zip(distances, rounds_list)
        println("\n>>> Distance $d, Rounds $r")

        # Create code and tanner graph
        code = SurfaceCode(d, d)
        tanner = CSSTannerGraph(code)

        for p_error in p_errors
            for p_loss in p_loss_rates
                config = AtomLossConfig(
                    d, r, p_error, p_meas,
                    p_loss,      # Single-qubit loss
                    p_loss * 2,  # Two-qubit loss (typically higher)
                    num_shots
                )

                prefix = @sprintf("atomloss_d%d_r%d_p%.4f_loss%.4f", d, r, p_error, p_loss)

                @printf("  Generating: p=%.4f, p_loss=%.4f ... ", p_error, p_loss)

                # Sample syndromes
                local_seed = SEED + abs(Int(hash((d, p_error, p_loss)) % 1000000))
                detection_events, observable_flips, loss_masks = sample_with_atom_loss(
                    tanner, config; seed=local_seed
                )

                # Save dataset
                metadata = save_dataset(OUTPUT_DIR, prefix, config,
                                       detection_events, observable_flips, loss_masks)

                push!(all_metadata, metadata)

                @printf("LER=%.4f, avg_loss=%.4f\n",
                       metadata["logical_error_rate"],
                       metadata["avg_loss_rate"])
            end
        end
    end

    # Save combined index
    index = Dict(
        "datasets" => all_metadata,
        "generator" => "TensorQEC atom loss model",
        "seed" => SEED
    )
    index_path = joinpath(OUTPUT_DIR, "atomloss_datasets_index.json")
    open(index_path, "w") do f
        JSON.print(f, index, 2)
    end

    println("\n" * "=" ^ 60)
    println("Dataset generation complete!")
    println("Generated $(length(all_metadata)) datasets")
    println("Output directory: $OUTPUT_DIR")
    println("=" ^ 60)
end

main()
