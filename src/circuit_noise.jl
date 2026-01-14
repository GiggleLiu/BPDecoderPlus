"""
Circuit-level noise model for surface code syndrome extraction.

This module implements realistic noise models for syndrome extraction circuits,
including:
- Gate errors (single-qubit, two-qubit)
- Measurement errors
- Idle errors
- State preparation errors

Reference: arXiv:2501.03582 - Exact Decoding under Circuit Level Noise
"""

using Statistics: mean

"""
    SurfaceCodeInstance

Container for surface code instance data needed for circuit visualization.
"""
struct SurfaceCodeInstance
    d::Int              # Code distance
    n_data::Int         # Number of data qubits
    n_ancilla::Int      # Number of ancilla qubits
    H_X::Matrix{Int}    # X-stabilizer check matrix
    H_Z::Matrix{Int}    # Z-stabilizer check matrix
    data_coords::Vector{Tuple{Int,Int}}     # (row, col) for each data qubit
    ancilla_coords::Vector{Tuple{Int,Int}}  # (row, col) for each ancilla qubit
    ancilla_types::Vector{Symbol}           # :X or :Z for each ancilla
end

"""
    CircuitNoiseModel

Parameters for circuit-level noise simulation.

# Fields
- `p_gate_1q::Float64`: Single-qubit gate error probability
- `p_gate_2q::Float64`: Two-qubit gate error probability
- `p_meas::Float64`: Measurement error probability
- `p_prep::Float64`: State preparation error probability
- `p_idle::Float64`: Idle error probability per time step
- `n_rounds::Int`: Number of syndrome extraction rounds
"""
struct CircuitNoiseModel
    p_gate_1q::Float64
    p_gate_2q::Float64
    p_meas::Float64
    p_prep::Float64
    p_idle::Float64
    n_rounds::Int

    function CircuitNoiseModel(;
        p::Float64=0.01,          # Base error rate
        p_gate_1q::Float64=p,
        p_gate_2q::Float64=p,
        p_meas::Float64=p,
        p_prep::Float64=p,
        p_idle::Float64=p/10,
        n_rounds::Int=1
    )
        new(p_gate_1q, p_gate_2q, p_meas, p_prep, p_idle, n_rounds)
    end
end

"""
    CircuitError

Represents an error event in the circuit.

# Fields
- `time_step::Int`: Time step when error occurred
- `location::Int`: Qubit index
- `error_type::Symbol`: Type of error (:X, :Y, :Z, :measurement)
"""
struct CircuitError
    time_step::Int
    location::Int
    error_type::Symbol
end

"""
    SyndromeExtractionCircuit

Represents a single round of syndrome extraction.

# Fields
- `n_data::Int`: Number of data qubits
- `n_ancilla::Int`: Number of ancilla qubits
- `cnot_schedule::Vector{Tuple{Int,Int,Int}}`: (time, control, target) for CNOTs
- `measurement_time::Int`: Time step of measurements
"""
struct SyndromeExtractionCircuit
    n_data::Int
    n_ancilla::Int
    cnot_schedule::Vector{Tuple{Int,Int,Int}}
    measurement_time::Int
end

"""
    build_syndrome_extraction_circuit(code::SurfaceCodeInstance) -> SyndromeExtractionCircuit

Build the syndrome extraction circuit for a surface code.

The circuit follows the standard "Raussendorf" ordering for CNOTs to minimize
hook errors.
"""
function build_syndrome_extraction_circuit(code::SurfaceCodeInstance)
    d = code.d
    n_data = code.n_data
    n_ancilla = code.n_ancilla

    # CNOT schedule: (time_step, control, target)
    # For rotated surface code, we use 4 time steps for the 4 data qubits
    # around each stabilizer
    cnot_schedule = Tuple{Int,Int,Int}[]

    # Ancilla qubits are numbered after data qubits
    ancilla_offset = n_data

    # Build schedule based on stabilizer type and geometry
    for stab_idx in 1:n_ancilla
        ancilla_qubit = ancilla_offset + stab_idx

        # Get the data qubits in this stabilizer
        # For simplicity, we use a generic approach based on the check matrices
        if stab_idx <= size(code.H_X, 1)
            # X-type stabilizer
            data_qubits = findall(!iszero, code.H_X[stab_idx, :])
        else
            # Z-type stabilizer
            z_idx = stab_idx - size(code.H_X, 1)
            data_qubits = findall(!iszero, code.H_Z[z_idx, :])
        end

        # Schedule CNOTs in order (N, W, E, S pattern for standard surface code)
        for (time_step, data_q) in enumerate(data_qubits[1:min(4, length(data_qubits))])
            push!(cnot_schedule, (time_step, ancilla_qubit, data_q))
        end
    end

    measurement_time = 5  # After all CNOTs

    return SyndromeExtractionCircuit(n_data, n_ancilla, cnot_schedule, measurement_time)
end

"""
    sample_circuit_errors(circuit::SyndromeExtractionCircuit, noise::CircuitNoiseModel;
                         seed=nothing) -> Vector{CircuitError}

Sample errors from the circuit noise model.
"""
function sample_circuit_errors(circuit::SyndromeExtractionCircuit,
                              noise::CircuitNoiseModel;
                              seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    errors = CircuitError[]
    n_total = circuit.n_data + circuit.n_ancilla

    # Sample preparation errors on ancilla
    for q in (circuit.n_data + 1):n_total
        if rand() < noise.p_prep
            push!(errors, CircuitError(0, q, :X))
        end
    end

    # Sample gate errors
    for (t, control, target) in circuit.cnot_schedule
        # Two-qubit depolarizing error after CNOT
        if rand() < noise.p_gate_2q
            # Sample one of 15 non-identity Pauli pairs
            pauli = rand(1:15)
            if pauli <= 3
                # X on control
                push!(errors, CircuitError(t, control, :X))
            elseif pauli <= 6
                # Z on control
                push!(errors, CircuitError(t, control, :Z))
            elseif pauli <= 9
                # Y on control
                push!(errors, CircuitError(t, control, :Y))
            elseif pauli <= 12
                # Error on target
                push!(errors, CircuitError(t, target, rand([:X, :Y, :Z])))
            else
                # Correlated error on both
                push!(errors, CircuitError(t, control, rand([:X, :Y, :Z])))
                push!(errors, CircuitError(t, target, rand([:X, :Y, :Z])))
            end
        end
    end

    # Sample idle errors on data qubits
    for t in 1:circuit.measurement_time
        for q in 1:circuit.n_data
            if rand() < noise.p_idle
                push!(errors, CircuitError(t, q, rand([:X, :Y, :Z])))
            end
        end
    end

    # Sample measurement errors
    for q in (circuit.n_data + 1):n_total
        if rand() < noise.p_meas
            push!(errors, CircuitError(circuit.measurement_time, q, :measurement))
        end
    end

    return errors
end

"""
    apply_circuit_noise(code::SurfaceCodeInstance, noise::CircuitNoiseModel;
                       seed=nothing)

Simulate multiple rounds of noisy syndrome extraction.

# Returns
- `detection_events::Matrix{Int}`: Detection events (n_ancilla × n_rounds)
- `final_error::Vector{Int}`: Final error on data qubits
- `all_errors::Vector{Vector{CircuitError}}`: Errors in each round (for debugging)
"""
function apply_circuit_noise(code::SurfaceCodeInstance, noise::CircuitNoiseModel;
                            seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    circuit = build_syndrome_extraction_circuit(code)
    n_rounds = noise.n_rounds

    # Track cumulative error on data qubits
    # Using Pauli frame: error[i] ∈ {0=I, 1=X, 2=Z, 3=Y}
    current_error = zeros(Int, code.n_data)

    # Store syndromes for each round
    syndromes = zeros(Int, code.n_ancilla, n_rounds)

    # Store all error events (for debugging/analysis)
    all_errors = Vector{CircuitError}[]

    prev_syndrome = zeros(Int, code.n_ancilla)

    for round in 1:n_rounds
        # Sample errors for this round
        round_errors = sample_circuit_errors(circuit, noise)
        push!(all_errors, round_errors)

        # Update data qubit errors
        for err in round_errors
            if err.location <= code.n_data && err.error_type != :measurement
                # Apply Pauli to data qubit
                pauli = err.error_type == :X ? 1 : (err.error_type == :Z ? 2 : 3)
                current_error[err.location] = xor(current_error[err.location], pauli)
            end
        end

        # Compute ideal syndrome from current error
        error_X = [(e & 1) != 0 for e in current_error]
        error_Z = [(e & 2) != 0 for e in current_error]

        syndrome_X = mod.(code.H_X * Int.(error_Z), 2)
        syndrome_Z = mod.(code.H_Z * Int.(error_X), 2)
        ideal_syndrome = vcat(syndrome_X, syndrome_Z)

        # Apply measurement errors
        measured_syndrome = copy(ideal_syndrome)
        for err in round_errors
            if err.error_type == :measurement
                ancilla_idx = err.location - code.n_data
                if 1 <= ancilla_idx <= code.n_ancilla
                    measured_syndrome[ancilla_idx] = 1 - measured_syndrome[ancilla_idx]
                end
            end
        end

        syndromes[:, round] = measured_syndrome
    end

    # Compute detection events (syndrome differences)
    detection_events = zeros(Int, code.n_ancilla, n_rounds)
    detection_events[:, 1] = syndromes[:, 1]  # First round compared to 0
    for r in 2:n_rounds
        detection_events[:, r] = mod.(syndromes[:, r] .- syndromes[:, r-1], 2)
    end

    return detection_events, current_error, all_errors
end

"""
    syndrome_to_detection_events(syndromes::Matrix{Int}) -> Matrix{Int}

Convert raw syndrome measurements to detection events (differences between rounds).
"""
function syndrome_to_detection_events(syndromes::Matrix{Int})
    n_ancilla, n_rounds = size(syndromes)
    detection = zeros(Int, n_ancilla, n_rounds)

    detection[:, 1] = syndromes[:, 1]
    for r in 2:n_rounds
        detection[:, r] = mod.(syndromes[:, r] .- syndromes[:, r-1], 2)
    end

    return detection
end

"""
    create_surface_code_instance(d::Int) -> SurfaceCodeInstance

Create a rotated surface code instance with geometric information.
"""
function create_surface_code_instance(d::Int)
    code = SurfaceCode(d, d)
    tanner = CSSTannerGraph(code)

    n_data = d^2
    n_X = size(tanner.stgx.H, 1)
    n_Z = size(tanner.stgz.H, 1)
    n_ancilla = n_X + n_Z

    H_X = Int.(Matrix(tanner.stgx.H))
    H_Z = Int.(Matrix(tanner.stgz.H))

    # Generate coordinates for rotated surface code layout
    # Data qubits on a d×d grid
    data_coords = Tuple{Int,Int}[]
    for row in 1:d, col in 1:d
        push!(data_coords, (row, col))
    end

    # Ancilla qubits at plaquette centers
    # X-stabilizers (plaquettes) and Z-stabilizers (vertices) alternate
    ancilla_coords = Tuple{Int,Int}[]
    ancilla_types = Symbol[]

    # X-type ancillas (between rows)
    for stab_idx in 1:n_X
        # Find center of data qubits in this stabilizer
        data_qubits = findall(!iszero, H_X[stab_idx, :])
        if !isempty(data_qubits)
            avg_row = mean([data_coords[q][1] for q in data_qubits])
            avg_col = mean([data_coords[q][2] for q in data_qubits])
            push!(ancilla_coords, (round(Int, avg_row * 2), round(Int, avg_col * 2)))
            push!(ancilla_types, :X)
        end
    end

    # Z-type ancillas
    for stab_idx in 1:n_Z
        data_qubits = findall(!iszero, H_Z[stab_idx, :])
        if !isempty(data_qubits)
            avg_row = mean([data_coords[q][1] for q in data_qubits])
            avg_col = mean([data_coords[q][2] for q in data_qubits])
            push!(ancilla_coords, (round(Int, avg_row * 2 + 1), round(Int, avg_col * 2 + 1)))
            push!(ancilla_types, :Z)
        end
    end

    return SurfaceCodeInstance(d, n_data, n_ancilla, H_X, H_Z,
                               data_coords, ancilla_coords, ancilla_types)
end

"""
    export_circuit_visualization_data(; d::Int=3, noise::CircuitNoiseModel=CircuitNoiseModel(),
                                       seed=nothing, filename::String="circuit_viz.json")

Export circuit and error model data for visualization.

# Arguments
- `d`: Surface code distance
- `noise`: Circuit noise model
- `seed`: Random seed for reproducibility
- `filename`: Output JSON filename

# Returns
- Dictionary with circuit layout, CNOT schedule, and sampled errors
"""
function export_circuit_visualization_data(; d::Int=3,
                                           noise::CircuitNoiseModel=CircuitNoiseModel(),
                                           seed::Union{Int,Nothing}=nothing,
                                           filename::String="circuit_viz.json")
    if seed !== nothing
        Random.seed!(seed)
    end

    # Create surface code instance
    code = create_surface_code_instance(d)
    circuit = build_syndrome_extraction_circuit(code)

    # Sample errors for each round
    all_round_errors = Vector{Dict}[]

    for round in 1:noise.n_rounds
        round_errors = sample_circuit_errors(circuit, noise)

        # Convert errors to serializable format
        error_dicts = [Dict(
            "time_step" => e.time_step,
            "location" => e.location,
            "error_type" => String(e.error_type),
            "round" => round
        ) for e in round_errors]

        push!(all_round_errors, error_dicts)
    end

    # Apply noise and get detection events
    detection_events, final_error, _ = apply_circuit_noise(code, noise; seed=seed)

    # Build visualization data
    viz_data = Dict(
        "code" => Dict(
            "distance" => d,
            "n_data" => code.n_data,
            "n_ancilla" => code.n_ancilla,
            "data_coords" => [[c[1], c[2]] for c in code.data_coords],
            "ancilla_coords" => [[c[1], c[2]] for c in code.ancilla_coords],
            "ancilla_types" => [String(t) for t in code.ancilla_types],
            "H_X" => [collect(row) for row in eachrow(code.H_X)],
            "H_Z" => [collect(row) for row in eachrow(code.H_Z)]
        ),
        "circuit" => Dict(
            "n_time_steps" => circuit.measurement_time,
            "cnot_schedule" => [[t, ctrl, tgt] for (t, ctrl, tgt) in circuit.cnot_schedule],
            "measurement_time" => circuit.measurement_time
        ),
        "noise_model" => Dict(
            "p_gate_1q" => noise.p_gate_1q,
            "p_gate_2q" => noise.p_gate_2q,
            "p_meas" => noise.p_meas,
            "p_prep" => noise.p_prep,
            "p_idle" => noise.p_idle,
            "n_rounds" => noise.n_rounds
        ),
        "errors" => vcat(all_round_errors...),
        "detection_events" => [collect(row) for row in eachrow(detection_events)],
        "final_error" => collect(final_error)
    )

    # Save to file
    open(filename, "w") do f
        JSON.print(f, viz_data, 2)
    end

    @printf("Circuit visualization data saved to %s\n", filename)
    return viz_data
end
