"""
Detector error model and hypergraph representation for decoding.

This module builds the detector graph (or hypergraph) that represents
the correlations between errors and detection events.

Reference: Stim documentation on detector error models
"""

"""
    DetectorGraph

Represents the detector error model as a weighted hypergraph.

# Fields
- `n_detectors::Int`: Number of detector nodes
- `n_logical::Int`: Number of logical observables
- `edges::Vector{HyperedgeData}`: Hyperedges (error mechanisms)
- `adjacency::Dict{Int, Vector{Int}}`: Detector -> edge indices
"""
struct DetectorGraph
    n_detectors::Int
    n_logical::Int
    edges::Vector{HyperedgeData}
    adjacency::Dict{Int, Vector{Int}}
end

"""
    HyperedgeData

Data for a single hyperedge in the detector graph.

# Fields
- `detectors::Vector{Int}`: Detector indices triggered by this error
- `logicals::Vector{Int}`: Logical observables flipped by this error
- `probability::Float64`: Error probability
- `weight::Float64`: Edge weight = -log(p/(1-p))
"""
struct HyperedgeData
    detectors::Vector{Int}
    logicals::Vector{Int}
    probability::Float64
    weight::Float64
end

"""
    build_detector_graph(code::SurfaceCodeInstance, noise::CircuitNoiseModel)
                        -> DetectorGraph

Build the detector error model from a surface code and noise model.

For code-capacity noise, this creates a simple graph where each data qubit
error connects two detectors.

For circuit-level noise, this creates a more complex hypergraph including
measurement errors and time-like correlations.
"""
function build_detector_graph(code::SurfaceCodeInstance, noise::CircuitNoiseModel)
    edges = HyperedgeData[]
    n_rounds = noise.n_rounds

    # Total detectors = n_ancilla × n_rounds + final check
    n_detectors = code.n_ancilla * n_rounds

    # For code-capacity model (single round), each data qubit error
    # affects the stabilizers it participates in

    if n_rounds == 1
        # Simple code-capacity noise model
        p = noise.p_gate_1q  # Use as overall error rate

        # X errors (detected by Z stabilizers)
        for q in 1:code.n_data
            affected_Z = findall(!iszero, code.H_Z[:, q])
            if !isempty(affected_Z)
                w = compute_weight(p)
                # Check if this error affects logical X
                logical = q in code.logical_X ? [1] : Int[]
                push!(edges, HyperedgeData(affected_Z, logical, p, w))
            end
        end

        # Z errors (detected by X stabilizers)
        for q in 1:code.n_data
            affected_X = findall(!iszero, code.H_X[:, q])
            if !isempty(affected_X)
                w = compute_weight(p)
                # Check if this error affects logical Z
                logical = q in code.logical_Z ? [1] : Int[]
                push!(edges, HyperedgeData(affected_X, logical, p, w))
            end
        end
    else
        # Circuit-level noise model
        build_circuit_level_edges!(edges, code, noise)
    end

    # Build adjacency map
    adjacency = Dict{Int, Vector{Int}}()
    for (edge_idx, edge) in enumerate(edges)
        for det in edge.detectors
            if !haskey(adjacency, det)
                adjacency[det] = Int[]
            end
            push!(adjacency[det], edge_idx)
        end
    end

    return DetectorGraph(n_detectors, 1, edges, adjacency)
end

"""
    build_circuit_level_edges!(edges, code, noise)

Add edges for circuit-level noise model including:
- Data qubit errors (space-like)
- Measurement errors (time-like)
- Hook errors from CNOT gates
"""
function build_circuit_level_edges!(edges::Vector{HyperedgeData},
                                   code::SurfaceCodeInstance,
                                   noise::CircuitNoiseModel)
    n_rounds = noise.n_rounds
    n_ancilla = code.n_ancilla

    # Helper to convert (round, ancilla) to detector index
    det_idx(r, a) = (r - 1) * n_ancilla + a

    for r in 1:n_rounds
        p = noise.p_gate_2q

        # Data qubit errors within a round
        for q in 1:code.n_data
            # X error
            affected_Z = findall(!iszero, code.H_Z[:, q])
            if !isempty(affected_Z)
                detectors = [det_idx(r, a) for a in affected_Z]
                logical = q in code.logical_X ? [1] : Int[]
                push!(edges, HyperedgeData(detectors, logical, p, compute_weight(p)))
            end

            # Z error
            affected_X = findall(!iszero, code.H_X[:, q])
            if !isempty(affected_X)
                detectors = [det_idx(r, a + size(code.H_Z, 1)) for a in affected_X]
                logical = q in code.logical_Z ? [1] : Int[]
                push!(edges, HyperedgeData(detectors, logical, p, compute_weight(p)))
            end
        end

        # Measurement errors (time-like edges)
        p_meas = noise.p_meas
        for a in 1:n_ancilla
            # Measurement error connects detector in round r and r+1
            if r < n_rounds
                detectors = [det_idx(r, a), det_idx(r + 1, a)]
            else
                # Last round - connects to final detector
                detectors = [det_idx(r, a)]
            end
            push!(edges, HyperedgeData(detectors, Int[], p_meas, compute_weight(p_meas)))
        end
    end
end

"""
    compute_weight(p::Float64) -> Float64

Compute the edge weight from error probability.
Weight = -log(p / (1-p)) for MWPM-style decoding.
"""
function compute_weight(p::Float64)
    if p <= 0 || p >= 1
        return Inf
    end
    return -log(p / (1 - p))
end

"""
    get_edge_subgraph(graph::DetectorGraph, active_detectors::Vector{Int})
                     -> DetectorGraph

Extract a subgraph containing only edges relevant to a set of active detectors.
"""
function get_edge_subgraph(graph::DetectorGraph, active_detectors::Set{Int})
    relevant_edges = HyperedgeData[]

    for edge in graph.edges
        if any(d in active_detectors for d in edge.detectors)
            push!(relevant_edges, edge)
        end
    end

    # Rebuild adjacency
    adjacency = Dict{Int, Vector{Int}}()
    for (edge_idx, edge) in enumerate(relevant_edges)
        for det in edge.detectors
            if !haskey(adjacency, det)
                adjacency[det] = Int[]
            end
            push!(adjacency[det], edge_idx)
        end
    end

    return DetectorGraph(graph.n_detectors, graph.n_logical, relevant_edges, adjacency)
end

"""
    detection_events_to_graph_nodes(detection_events::Matrix{Int}) -> Vector{Int}

Convert detection event matrix to list of triggered detector indices.
"""
function detection_events_to_graph_nodes(detection_events::Matrix{Int})
    n_ancilla, n_rounds = size(detection_events)
    triggered = Int[]

    for r in 1:n_rounds
        for a in 1:n_ancilla
            if detection_events[a, r] == 1
                det_idx = (r - 1) * n_ancilla + a
                push!(triggered, det_idx)
            end
        end
    end

    return triggered
end

"""
    build_tanner_graph(code::SurfaceCodeInstance) -> SimpleDiGraph

Build the Tanner graph (bipartite graph) for belief propagation.

# Returns
A bipartite graph where:
- Variable nodes 1:n_data represent data qubits
- Check nodes (n_data+1):(n_data+n_checks) represent stabilizers
"""
function build_tanner_graph(code::SurfaceCodeInstance)
    n_data = code.n_data
    n_X_checks = size(code.H_X, 1)
    n_Z_checks = size(code.H_Z, 1)
    n_checks = n_X_checks + n_Z_checks

    n_total = n_data + n_checks
    g = SimpleDiGraph(n_total)

    # Add edges from X-type checks
    for c in 1:n_X_checks
        for v in findall(!iszero, code.H_X[c, :])
            add_edge!(g, n_data + c, v)  # Check -> Variable
            add_edge!(g, v, n_data + c)  # Variable -> Check
        end
    end

    # Add edges from Z-type checks
    for c in 1:n_Z_checks
        for v in findall(!iszero, code.H_Z[c, :])
            add_edge!(g, n_data + n_X_checks + c, v)
            add_edge!(g, v, n_data + n_X_checks + c)
        end
    end

    return g
end
