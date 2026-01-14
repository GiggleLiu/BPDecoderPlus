#!/usr/bin/env julia
"""
Generate benchmark data for decoder comparison.

This script generates test scenarios for comparing IP and BP decoders
at various code distances and error rates.

Usage:
    julia benchmark/generate_data.jl [--quick]
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using BPDecoderPlus
using Random
using JSON
using Printf

# Configuration
const QUICK_MODE = "--quick" in ARGS

if QUICK_MODE
    println("Running in quick mode (fewer trials)...")
    const DISTANCES = [3, 5]
    const ERROR_RATES = [0.02, 0.05, 0.08, 0.10]
    const N_TRIALS = 100
else
    const DISTANCES = [3, 5, 7, 9]
    const ERROR_RATES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
    const N_TRIALS = 1000
end

const OUTPUT_DIR = joinpath(dirname(@__FILE__), "data")
const SEED = 42

function main()
    # Ensure output directory exists
    mkpath(OUTPUT_DIR)

    println("="^60)
    println("BPDecoderPlus Benchmark Data Generation")
    println("="^60)
    println("Distances: ", DISTANCES)
    println("Error rates: ", ERROR_RATES)
    println("Trials per point: ", N_TRIALS)
    println("Output directory: ", OUTPUT_DIR)
    println("="^60)

    # Set random seed for reproducibility
    Random.seed!(SEED)

    # Generate data for IP decoder (baseline)
    println("\n>>> Generating IP decoder data...")
    ip_results = run_threshold_simulation(
        DISTANCES, ERROR_RATES, N_TRIALS;
        decoder=:IP, seed=SEED
    )
    ip_file = joinpath(OUTPUT_DIR, "ip_decoder_results.json")
    save_results(ip_results, ip_file)

    # Generate data for BP decoder (without OSD)
    println("\n>>> Generating BP decoder data...")
    bp_results = run_threshold_simulation(
        DISTANCES, ERROR_RATES, N_TRIALS;
        decoder=:BP, seed=SEED
    )
    bp_file = joinpath(OUTPUT_DIR, "bp_decoder_results.json")
    save_results(bp_results, bp_file)

    # Generate data for BP+OSD decoder
    println("\n>>> Generating BP+OSD decoder data...")
    bposd_results = run_threshold_simulation(
        DISTANCES, ERROR_RATES, N_TRIALS;
        decoder=:BPOSD, seed=SEED
    )
    bposd_file = joinpath(OUTPUT_DIR, "bposd_decoder_results.json")
    save_results(bposd_results, bposd_file)

    # Generate atom loss data
    println("\n>>> Generating atom loss data...")
    generate_atom_loss_data()

    println("\n", "="^60)
    println("Data generation complete!")
    println("Results saved to: ", OUTPUT_DIR)
    println("="^60)
end

function generate_atom_loss_data()
    """Generate benchmark data for atom loss scenarios."""
    loss_rates = [0.0, 0.01, 0.02, 0.03]
    distances = QUICK_MODE ? [3, 5] : [3, 5, 7]
    n_trials = QUICK_MODE ? 100 : 500
    p_error = 0.05  # Fixed error rate

    results = Dict{String,Any}(
        "loss_rates" => loss_rates,
        "distances" => distances,
        "p_error" => p_error,
        "n_trials" => n_trials,
        "data" => Dict{String,Any}()
    )

    for d in distances
        println("  Distance $d:")
        code = SurfaceCode(d, d)
        tanner = CSSTannerGraph(code)
        n_qubits = d^2
        em = iid_error(p_error/3, p_error/3, p_error/3, n_qubits)

        results["data"][string(d)] = Dict{String,Any}()

        for p_loss in loss_rates
            n_errors = 0
            loss_model = AtomLossModel(p_loss)
            decoder = IPDecoder()
            compiled = compile(decoder, tanner)

            for _ in 1:n_trials
                # Apply atom loss
                _, lost_qubits = apply_atom_loss(tanner, loss_model)

                # Sample error
                ep = random_error_pattern(em)

                # Get syndrome
                syn = syndrome_extraction(ep, tanner)

                # Decode (with or without loss information)
                result = decode(compiled, syn)

                # Check logical error
                if has_logical_error(tanner, ep, result.error_pattern)
                    n_errors += 1
                end
            end

            ler = n_errors / n_trials
            std_err = sqrt(ler * (1 - ler) / n_trials)

            results["data"][string(d)][string(p_loss)] = Dict(
                "logical_error_rate" => ler,
                "std_error" => std_err
            )

            Printf.@printf("    p_loss=%.2f: LER=%.4f +/- %.4f\n", p_loss, ler, std_err)
        end
    end

    output_file = joinpath(OUTPUT_DIR, "atom_loss_results.json")
    save_results(results, output_file)
end

# Run main function
main()
