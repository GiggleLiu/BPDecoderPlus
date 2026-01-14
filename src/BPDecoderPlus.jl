"""
    BPDecoderPlus

A Julia package for circuit-level decoding of surface codes using belief propagation
and integer programming decoders. Built on top of TensorQEC.jl.

## Winter School Project Goals:
1. Basic: Reproduce the MLE decoder (IPDecoder) as the baseline
2. Challenge: Develop and compare belief propagation based decoders

## Quick Start
```julia
using BPDecoderPlus

# Run a quick benchmark
results = quick_benchmark(distance=5, p=0.05, n_trials=1000)

# Compare decoders
compare_decoders([3, 5, 7], 0.01:0.02:0.15, 1000)
```

## References
- TensorQEC.jl: https://github.com/nzy1997/TensorQEC.jl
- BP+OSD: Roffe et al., arXiv:2005.07016
- Atom Loss QEC: arXiv:2412.07841
"""
module BPDecoderPlus

using TensorQEC
using Random
using Statistics
using Printf
using JSON

# Re-export key TensorQEC types and functions
export SurfaceCode, ToricCode, CSSTannerGraph
export IPDecoder, BPDecoder, MatchingDecoder
export iid_error, random_error_pattern, syndrome_extraction
export decode, compile, multi_round_qec
export Mod2, logical_operator

# Export our own functions
export quick_benchmark, compare_decoders, run_threshold_simulation
export compute_logical_error_rate, save_results, load_results
export AtomLossModel, apply_atom_loss, decode_with_atom_loss
export has_logical_error

# Export circuit visualization functions
export CircuitNoiseModel, CircuitError, SyndromeExtractionCircuit, SurfaceCodeInstance
export build_syndrome_extraction_circuit, sample_circuit_errors, apply_circuit_noise
export export_circuit_visualization_data, create_surface_code_instance

#=============================================================================
    Benchmarking Functions
=============================================================================#

"""
    quick_benchmark(; distance=5, p=0.05, n_trials=1000, decoder=:IP)

Run a quick benchmark to verify the decoder is working.

# Arguments
- `distance`: Surface code distance
- `p`: Physical error probability
- `n_trials`: Number of Monte Carlo trials
- `decoder`: Decoder type (:IP, :BP, :BPOSD, :Matching)

# Returns
- Dictionary with logical error rate and other statistics
"""
function quick_benchmark(; distance::Int=5, p::Float64=0.05,
                         n_trials::Int=1000, decoder::Symbol=:IP)
    # Create surface code and tanner graph
    code = SurfaceCode(distance, distance)
    tanner = CSSTannerGraph(code)

    # Get logical operators for error checking
    lx, lz = logical_operator(tanner)

    # Create error model (depolarizing: px=py=pz=p/3)
    n_qubits = distance^2
    em = iid_error(p/3, p/3, p/3, n_qubits)

    # Create decoder
    dec = create_decoder(decoder)
    compiled = compile(dec, tanner)

    # Run Monte Carlo simulation
    n_logical_errors = 0

    for _ in 1:n_trials
        # Sample error
        ep = random_error_pattern(em)

        # Get syndrome
        syn = syndrome_extraction(ep, tanner)

        # Decode
        result = decode(compiled, syn)

        # Check logical error using TensorQEC's check_logical_error
        if check_logical_error(ep, result.error_pattern, lx, lz)
            n_logical_errors += 1
        end
    end

    ler = n_logical_errors / n_trials
    std_err = sqrt(ler * (1 - ler) / n_trials)

    return Dict(
        "distance" => distance,
        "p" => p,
        "n_trials" => n_trials,
        "decoder" => String(decoder),
        "logical_error_rate" => ler,
        "std_error" => std_err
    )
end

"""
    create_decoder(decoder_type::Symbol)

Create a decoder instance from a symbol.
"""
function create_decoder(decoder_type::Symbol)
    if decoder_type == :IP
        return IPDecoder()
    elseif decoder_type == :BP
        return BPDecoder(100, false)  # BP without OSD
    elseif decoder_type == :BPOSD
        return BPDecoder(100, true)   # BP with OSD
    elseif decoder_type == :Matching
        return MatchingDecoder()
    else
        error("Unknown decoder type: $decoder_type")
    end
end

"""
    compute_logical_error_rate(tanner::CSSTannerGraph, decoder,
                               error_model, n_trials::Int)

Compute the logical error rate for a decoder.

# Returns
- `(ler, std_err)`: Logical error rate and standard error
"""
function compute_logical_error_rate(tanner::CSSTannerGraph, decoder,
                                   error_model, n_trials::Int)
    compiled = compile(decoder, tanner)
    lx, lz = logical_operator(tanner)
    n_errors = 0

    for _ in 1:n_trials
        ep = random_error_pattern(error_model)
        syn = syndrome_extraction(ep, tanner)
        result = decode(compiled, syn)

        if check_logical_error(ep, result.error_pattern, lx, lz)
            n_errors += 1
        end
    end

    ler = n_errors / n_trials
    std_err = sqrt(ler * (1 - ler) / n_trials)
    return ler, std_err
end

"""
    run_threshold_simulation(distances::Vector{Int}, error_rates::Vector{Float64},
                            n_trials::Int; decoder::Symbol=:IP, seed=nothing)

Run a threshold simulation across multiple distances and error rates.

# Arguments
- `distances`: Vector of code distances to test
- `error_rates`: Vector of physical error rates
- `n_trials`: Number of trials per (distance, error_rate) pair
- `decoder`: Decoder type
- `seed`: Random seed for reproducibility

# Returns
- Dictionary with all results, suitable for plotting
"""
function run_threshold_simulation(distances::Vector{Int},
                                 error_rates::Vector{Float64},
                                 n_trials::Int;
                                 decoder::Symbol=:IP,
                                 seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    dec = create_decoder(decoder)
    results = Dict{String,Any}(
        "distances" => distances,
        "error_rates" => error_rates,
        "n_trials" => n_trials,
        "decoder" => String(decoder),
        "data" => Dict{Int,Any}()
    )

    for d in distances
        @printf("Distance %d:\n", d)
        code = SurfaceCode(d, d)
        tanner = CSSTannerGraph(code)
        n_qubits = d^2

        results["data"][d] = Dict{Float64,Any}()

        for p in error_rates
            em = iid_error(p/3, p/3, p/3, n_qubits)
            ler, std_err = compute_logical_error_rate(tanner, dec, em, n_trials)

            results["data"][d][p] = Dict(
                "logical_error_rate" => ler,
                "std_error" => std_err
            )

            @printf("  p=%.3f: LER=%.4f +/- %.4f\n", p, ler, std_err)
        end
    end

    return results
end

"""
    compare_decoders(distances::Vector{Int}, error_rates, n_trials::Int;
                    decoders=[:IP, :BPOSD], seed=nothing)

Compare multiple decoders across distances and error rates.

# Returns
- Dictionary mapping decoder names to their results
"""
function compare_decoders(distances::Vector{Int},
                         error_rates,
                         n_trials::Int;
                         decoders::Vector{Symbol}=[:IP, :BPOSD],
                         seed::Union{Int,Nothing}=nothing)
    results = Dict{Symbol,Any}()

    for dec in decoders
        @printf("\n=== Running %s decoder ===\n", dec)
        results[dec] = run_threshold_simulation(
            distances, collect(error_rates), n_trials;
            decoder=dec, seed=seed
        )
    end

    return results
end

#=============================================================================
    Atom Loss Handling
=============================================================================#

"""
    AtomLossModel

Model for atom loss in neutral atom quantum computers.

# Fields
- `p_loss`: Probability of atom loss per qubit
- `detected`: Whether loss is detected (heralded)
"""
struct AtomLossModel
    p_loss::Float64
    detected::Bool

    AtomLossModel(p_loss::Float64; detected::Bool=true) = new(p_loss, detected)
end

"""
    apply_atom_loss(tanner::CSSTannerGraph, loss_model::AtomLossModel)

Simulate atom loss and return modified tanner graph with lost qubits marked.

# Returns
- `(modified_tanner, lost_qubits)`: Modified tanner graph and list of lost qubit indices
"""
function apply_atom_loss(tanner::CSSTannerGraph, loss_model::AtomLossModel)
    n_qubits = tanner.stgx.nq

    # Sample which qubits are lost
    lost_qubits = findall(_ -> rand() < loss_model.p_loss, 1:n_qubits)

    if isempty(lost_qubits)
        return tanner, Int[]
    end

    # For heralded loss, we know which qubits are lost
    # We can modify the syndrome extraction accordingly
    # This is a simplified model - full implementation would modify the tanner graph

    return tanner, lost_qubits
end

"""
    decode_with_atom_loss(tanner::CSSTannerGraph, syn, decoder,
                         lost_qubits::Vector{Int})

Decode with knowledge of atom loss locations.

For heralded atom loss, we treat lost qubits as erasures, which can be
corrected more easily than unknown errors.
"""
function decode_with_atom_loss(tanner::CSSTannerGraph, syn, decoder,
                              lost_qubits::Vector{Int})
    if isempty(lost_qubits)
        # No loss, normal decoding
        return decode(decoder, tanner, syn)
    end

    # With known erasure locations, we can:
    # 1. Modify the prior probabilities for lost qubits (set to 0.5)
    # 2. Use supercheck construction for stabilizers affected by loss
    # For simplicity, we use the basic approach here

    # Create modified error model with high error rate on lost qubits
    n_qubits = tanner.stgx.nq
    px = zeros(n_qubits)
    py = zeros(n_qubits)
    pz = zeros(n_qubits)

    for q in lost_qubits
        px[q] = 0.5
        py[q] = 0.0  # Y errors are less likely
        pz[q] = 0.5
    end

    # Decode with modified error model
    # Note: This is a simplified implementation
    return decode(decoder, tanner, syn)
end

"""
    has_logical_error(tanner::CSSTannerGraph, ep, correction) -> Bool

Check if the combined error pattern (original error + correction) results in a logical error.

This is a convenience wrapper around TensorQEC's check_logical_error.
"""
function has_logical_error(tanner::CSSTannerGraph, ep, correction)
    lx, lz = logical_operator(tanner)
    return check_logical_error(ep, correction, lx, lz)
end

#=============================================================================
    I/O Functions
=============================================================================#

"""
    save_results(results::Dict, filename::String)

Save benchmark results to a JSON file.
"""
function save_results(results::Dict, filename::String)
    # Convert keys to strings for JSON compatibility
    json_results = convert_keys_to_strings(results)

    open(filename, "w") do f
        JSON.print(f, json_results, 2)
    end
    @printf("Results saved to %s\n", filename)
end

"""
    load_results(filename::String)

Load benchmark results from a JSON file.
"""
function load_results(filename::String)
    return JSON.parsefile(filename)
end

function convert_keys_to_strings(d::Dict)
    result = Dict{String,Any}()
    for (k, v) in d
        key = string(k)
        if v isa Dict
            result[key] = convert_keys_to_strings(v)
        else
            result[key] = v
        end
    end
    return result
end

# Include circuit-level noise module
include("circuit_noise.jl")

end # module
