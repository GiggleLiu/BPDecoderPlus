#!/usr/bin/env julia
"""
Run decoder benchmarks and timing comparisons.

This script compares the performance (accuracy and speed) of different decoders.

Usage:
    julia benchmark/run_benchmarks.jl [--quick]
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using BPDecoderPlus
using Random
using Statistics
using Printf
using JSON

const QUICK_MODE = "--quick" in ARGS
const OUTPUT_DIR = joinpath(dirname(@__FILE__), "data")

function main()
    mkpath(OUTPUT_DIR)

    println("="^60)
    println("BPDecoderPlus Decoder Benchmarks")
    println("="^60)

    # 1. Quick accuracy test
    println("\n>>> Quick Accuracy Test")
    test_decoder_accuracy()

    # 2. Timing benchmark
    println("\n>>> Timing Benchmark")
    timing_results = benchmark_decoder_timing()
    save_results(timing_results, joinpath(OUTPUT_DIR, "timing_results.json"))

    # 3. Scalability test
    println("\n>>> Scalability Test")
    scalability_results = benchmark_scalability()
    save_results(scalability_results, joinpath(OUTPUT_DIR, "scalability_results.json"))

    println("\n", "="^60)
    println("Benchmarks complete!")
    println("="^60)
end

function test_decoder_accuracy()
    """Quick test to verify decoders are working correctly."""
    distance = 5
    p = 0.05
    n_trials = QUICK_MODE ? 100 : 500

    println("  Testing decoders on d=$distance, p=$p, n=$n_trials trials")

    for decoder_type in [:IP, :BP, :BPOSD]
        result = quick_benchmark(
            distance=distance,
            p=p,
            n_trials=n_trials,
            decoder=decoder_type
        )
        @printf("  %6s: LER=%.4f +/- %.4f\n",
                decoder_type,
                result["logical_error_rate"],
                result["std_error"])
    end
end

function benchmark_decoder_timing()
    """Benchmark decoder execution time."""
    distances = QUICK_MODE ? [3, 5, 7] : [3, 5, 7, 9, 11]
    n_trials = QUICK_MODE ? 50 : 200
    p = 0.05

    results = Dict{String,Any}(
        "distances" => distances,
        "n_trials" => n_trials,
        "p" => p,
        "data" => Dict{String,Any}()
    )

    for decoder_type in [:IP, :BP, :BPOSD]
        println("  Timing $decoder_type decoder...")
        results["data"][string(decoder_type)] = Dict{String,Any}()

        for d in distances
            code = SurfaceCode(d, d)
            tanner = CSSTannerGraph(code)
            n_qubits = d^2
            em = iid_error(p/3, p/3, p/3, n_qubits)

            dec = create_decoder(decoder_type)
            compiled = compile(dec, tanner)

            # Warm-up
            ep = random_error_pattern(em)
            syn = syndrome_extraction(ep, tanner)
            _ = decode(compiled, syn)

            # Timing
            times = Float64[]
            for _ in 1:n_trials
                ep = random_error_pattern(em)
                syn = syndrome_extraction(ep, tanner)

                t_start = time()
                _ = decode(compiled, syn)
                t_end = time()

                push!(times, (t_end - t_start) * 1000)  # Convert to ms
            end

            mean_time = mean(times)
            std_time = std(times)

            results["data"][string(decoder_type)][string(d)] = Dict(
                "mean_time_ms" => mean_time,
                "std_time_ms" => std_time,
                "min_time_ms" => minimum(times),
                "max_time_ms" => maximum(times)
            )

            @printf("    d=%2d: %.3f +/- %.3f ms\n", d, mean_time, std_time)
        end
    end

    return results
end

function benchmark_scalability()
    """Test how decoder performance scales with code distance."""
    distances = QUICK_MODE ? [3, 5, 7, 9] : [3, 5, 7, 9, 11, 13, 15]
    n_trials = QUICK_MODE ? 100 : 500
    p = 0.05

    results = Dict{String,Any}(
        "distances" => distances,
        "n_trials" => n_trials,
        "p" => p,
        "data" => Dict{String,Any}()
    )

    # Test IP decoder scalability
    println("  Testing IP decoder scalability...")
    results["data"]["IP"] = Dict{String,Any}()

    for d in distances
        code = SurfaceCode(d, d)
        tanner = CSSTannerGraph(code)
        n_qubits = d^2
        em = iid_error(p/3, p/3, p/3, n_qubits)

        dec = IPDecoder()
        compiled = compile(dec, tanner)

        n_errors = 0
        total_time = 0.0

        for _ in 1:n_trials
            ep = random_error_pattern(em)
            syn = syndrome_extraction(ep, tanner)

            t_start = time()
            result = decode(compiled, syn)
            t_end = time()

            total_time += (t_end - t_start)

            if has_logical_error(tanner, ep, result.error_pattern)
                n_errors += 1
            end
        end

        ler = n_errors / n_trials
        avg_time = total_time / n_trials * 1000

        results["data"]["IP"][string(d)] = Dict(
            "logical_error_rate" => ler,
            "avg_time_ms" => avg_time,
            "n_qubits" => n_qubits
        )

        @printf("    d=%2d (n=%3d): LER=%.4f, time=%.3f ms\n", d, n_qubits, ler, avg_time)
    end

    return results
end

function create_decoder(decoder_type::Symbol)
    if decoder_type == :IP
        return IPDecoder()
    elseif decoder_type == :BP
        return BPDecoder(100, false)
    elseif decoder_type == :BPOSD
        return BPDecoder(100, true)
    else
        error("Unknown decoder type: $decoder_type")
    end
end

# Run main
main()
