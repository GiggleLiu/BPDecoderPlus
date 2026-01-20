#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2cm),
)

#set text(
  font: ("Source Han Serif SC", "Linux Libertine"),
  size: 11pt,
  lang: "en"
)

#set par(
  justify: true,
  first-line-indent: 2em,
  leading: 0.8em
)

#show heading: it => [
  #v(0.5em)
  #text(weight: "bold", size: 1.1em, it)
  #v(0.3em)
]

= What is a probabilistic graphical model (PGM)?

A probabilistic graphical model (PGM) combines **probability theory** and **graph theory**. It uses a graph to represent conditional independence assumptions between random variables, allowing a compact description of a joint probability distribution.

Simply put:
- **Nodes**: represent random variables.
- **Edges**: represent probabilistic dependencies between variables.

With this representation, a complex joint distribution $P(X_1, X_2, dots, X_n)$ can be factorized into local factors, reducing computational and storage complexity.

== 1. Core categories

Based on edge direction, graphical models are mainly divided into two categories:

=== 1.1 Bayesian network
- **Structure**: Directed acyclic graph (DAG).
- **Meaning**: represents causal or explicit dependency relations.
- **Factorization**:
  The joint distribution factorizes into the product of each node's conditional probability given its parents.

  $
    P(X_1, dots, X_n) = product_(i=1)^n P(X_i | "pa"(X_i))
  $
  where $"pa"(X_i)$ is the parent set of node $X_i$.

- **Typical applications**: disease diagnosis (cause $arrow$ symptom), hidden Markov models (HMM).

=== 1.2 Markov random field (MRF)
- **Structure**: Undirected graph.
- **Meaning**: represents correlations or constraints among variables, without explicit direction (e.g., neighboring pixels in an image).
- #link(<HAMMERSLEY_CLIFFORD_THEOREM>)[#text(fill:red)[**Factorization**]]:
  Defined via #link(<MAXIMAL_CLIQUE>)[#text(fill:red)[maximal cliques]] and #link(<POTENTIAL_FUNCTION>)[#text(fill:red)[potential functions]].

  $
    P(X) = 1/Z product_(C in cal(C)) psi_C(X_C)
  $
  where:
  - $cal(C)$ is the set of all maximal cliques in the graph.
  - $psi_C(X_C)$ is the potential function on clique $C$, typically requiring $psi_C >= 0$.
  - $Z$ is the partition function for normalization:
    $ Z = sum_X product_(C in cal(C)) psi_C(X_C) $

- **Typical applications**: image segmentatiFon, conditional random fields (CRF).


== 2. The three basic problems in graphical models

In practice, we focus on three core problems:

=== 2.1 Representation
How do we choose the graph structure and parameters to model the real world?
- Choose directed or undirected graphs?
- Define which variables are conditionally independent?
- *Key idea*: use conditional independence $X perp Y | Z$ to sparsify the graph.

=== 2.2 Inference
Given observed variables $E$ (Evidence), compute the posterior $P(Y | E)$ of hidden variables $Y$.

Common algorithms:
- **Exact inference**:
  - Variable Elimination.
  - Clique Tree / Junction Tree Algorithm.
  - Belief Propagation (BP) (exact on trees).
- **Approximate inference** (for complex graphs):
  - Variational Inference: fit a simple distribution $Q(Y)$ to approximate $P(Y|E)$.
  - Monte Carlo sampling (MCMC): e.g., Gibbs sampling.

=== 2.3 Learning
Given a dataset, how do we learn the model?
- **Parameter learning**: with known graph structure, estimate parameters of CPTs or potentials (MLE or EM).
- **Structure learning**: both parameters and graph structure are unknown and must be inferred from data.

== 3. Example: a simple Bayesian network

Suppose we have three variables:
- $R$: Rain
- $S$: Sprinkler
- $G$: Grass Wet

Relationships:
1. Rain $R$ affects grass wetness $G$.
2. Sprinkler $S$ affects grass wetness $G$.
3. Assume $R$ and $S$ are independent (simplified).

Graph: $R arrow G, S arrow G$.

The joint distribution is:
$
  P(R, S, G) = P(R) dot P(S) dot P(G | R, S)
$

Compared to storing all $2^3=8$ combinations, we only store $P(R), P(S)$, and $P(G|R,S)$, greatly reducing parameter count.

== 4. Summary

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 10pt,
  align: horizon,
  stroke: none,
  table.header(
    [*Feature*], [*Bayesian network*], [*Markov network*]
  ),
  table.hline(stroke: 0.5pt),
  [Graph type], [Directed acyclic graph (DAG)], [Undirected graph],
  [Dependency], [Causal relation], [Correlation/spatial constraint],
  [Local factor], [Conditional probability $P(X|Y)$], [Potential $psi(X, Y)$],
  [Normalization], [Local normalization], [Global normalization ($Z$)]
)

// Part 2

== 2. Undirected probabilistic graph (Markov Random Field)

An undirected probabilistic graphical model, commonly called a *Markov random field (MRF)*, uses an undirected graph $G=(V, E)$ to describe the joint distribution of a set of random variables. Unlike Bayesian networks, MRF edges have no arrows, indicating correlation or constraints without explicit causal direction.

=== 2.1 Factorization of the joint distribution

As noted above, an MRF factorizes its joint distribution based on *maximal cliques*.

Let $cal(C)$ be the set of all maximal cliques. Let $X_C$ be the variables in clique $C$. The joint distribution is a Gibbs distribution:

$
  P(X) = 1/Z product_(C in cal(C)) psi_C(X_C)
$

where:
- $psi_C(X_C) >= 0$ are potentials (factors).
- $Z$ is the *partition function*, the hardest part of the model:

$
  Z = sum_(X) product_(C in cal(C)) psi_C(X_C)
$

#block(fill: luma(240), inset: 8pt, radius: 4pt)[
  *Note*: Computing $Z$ requires summing over #link(<PARTITION_FUNCTION>)[#text(fill:red)[all possible configurations.]] If $X$ has $n$ binary variables, the sum has $2^n$ terms, which is usually intractable (NP-hard) for large $n$.
]

=== 2.2 Markov properties (conditional independence)

In an undirected graph, connectivity directly defines conditional independence. There are three equivalent definitions (for strictly positive $P(X)>0$):

+ *Pairwise Markov property*
  Let $u$ and $v$ be two nodes with no edge between them. Given all other nodes $X_{V backslash {u,v}}$, $u$ and $v$ are independent.
  $
    X_u perp X_v mid(|) X_{V backslash {u,v}}
  $

+ *Local Markov property*
  A node $v$ is independent of all other nodes given its *neighbors* $N(v)$.
  $
    X_v perp X_{V backslash ({v} union N(v))} mid(|) X_{N(v)}
  $
  Here, $N(v)$ is the *Markov blanket* of $v$. In undirected graphs, it is just the neighbors; in directed graphs, it is more complex (parents, children, and parents of children).

+ *Global Markov property*
  The most intuitive definition. Let node sets $A$ and $B$ be *separated* by $S$ (all paths from $A$ to $B$ pass through $S$), then:
  $
    X_A perp X_B mid(|) X_S
  $

=== 2.3 Energy models and physics

MRFs are closely related to statistical physics. We can write potentials as exponentials:
$
  psi_C(X_C) = exp(-E_C(X_C))
$
where $E_C$ is an *energy function*.

Then the joint distribution becomes a Boltzmann distribution:
$
  P(X) = 1/Z exp(- sum_(C in cal(C)) E_C(X_C)) = 1/Z exp(-E_"total"(X))
$

This explains why we often say: *lower energy means higher probability*.

*Classic example: Ising model*
Used to model ferromagnetism. Nodes are arranged on a grid, each node $x_i in {+1, -1}$.
The energy function is:
$
  E(x) = - sum_((i,j) in E) J x_i x_j - sum_i h x_i
$
- The first term says neighboring spins prefer to align (if $J>0$).
- The second term captures the effect of an external field.
This is a classic pairwise MRF.

=== 2.4 Application scenarios

Because undirected graphs model "neighbor relations" well, they are widely used in:

- *Computer vision*:
  - Image segmentation: neighboring pixels prefer the same label.
  - Image denoising: observed pixels correlate with true pixels, and true pixels vary smoothly.
- *Natural language processing*:
  - Conditional random fields (CRF): sequence labeling (e.g., NER), more flexible than HMM for long-range dependencies.
- *Spatial statistics*:
  - Predict spatially correlated variables.

=== 2.5 Summary: directed vs undirected

#table(
  columns: (1fr, 3fr, 3fr),
  inset: 10pt,
  stroke: 0.5pt + gray,
  align: horizon,
  table.header([*Dimension*], [*Bayesian network (directed)*], [*Markov network (undirected)*]),
  [Factor definition], [Conditional probability $P(X|Y)$, locally normalized], [Potential $psi(X,Y)$, requires global normalization $Z$],
  [Independence], [Dependence direction (d-separation)], [Connectivity (graph separation)],
  [Applicability], [Causal inference, logical reasoning], [Images, spatial, constraint systems]
)


== 3. PGM perspective in QEC

Quantum error correction (especially topological codes) can be mapped perfectly to **inference on an undirected graphical model (MRF)**.

=== 3.1 Mapping: Tanner graph and factor graph

In QEC, we often use #link(<GRAPH_RELATIONSHIP>)[#text(fill:red)[Tanner graphs to describe codes. This is essentially the **factor graph** in a graphical model]]:

- **Variable nodes**: physical qubit error states.
  Let $E = {e_1, e_2, dots, e_n}$ be error chains, typically $e_i in {I, X, Y, Z}$. In the simplest model, we only consider bit flips, $e_i in {0, 1}$. This corresponds to hidden variables $X$.

- **Factor nodes**: stabilizer measurements or parity checks.
  These nodes define constraints between variables.

- **Evidence**: the syndrome $S$.
  When we measure stabilizers and violate a constraint (e.g., odd parity), we observe a non-zero syndrome.

=== 3.2 #link(<MAP>)[#text(fill:red)[From MAP to MMAP]]

Understanding why QEC is an MMAP problem hinges on the difference between **"most likely physical error"** and **"most likely logical error"**.

1.  **MAP - find a specific error**
    If we try to find the single most likely physical error chain $E^*$:
    $
      E^* = "argmax"_E P(E | S)
    $
    This corresponds to what *minimum weight perfect matching (MWPM)* solves (in non-degenerate cases): the shortest path explaining the syndrome.

2.  **MMAP - find a logical equivalence class**
    In QEC, we apply a correction $R$. If $R$ differs from the true error $E$ by a stabilizer $g$ (i.e., $R = E dot g$), the correction succeeds because stabilizers do not affect the logical state.

    This introduces **degeneracy**: *many different physical errors $E$ correspond to the same logical class*.

    We group all physical errors by their logical operator homology class $bar(L)$. What we care about is: *which logical class is most likely?*

    We sum probabilities over all physical errors in the same class, then take the maximum:

    $
      bar(L)^* = "argmax"_(bar(L)) sum_(E in bar(L)) P(E | S)
    $

    This is the standard **MMAP** problem:
    - **Marginal**: sum (marginalize) over physical details we do not care about.
    - **MAP**: maximize at the logical-class level.

=== 3.3 Physical meaning: Ising model and partition function

This mapping is physically elegant. For the surface code:
- The error model maps to a **random-bond Ising model**.
- Solving MMAP is equivalent to computing the **free energy** of this statistical system.

To compare two logical classes $bar(L)_1$ and $bar(L)_2$, we compute their partition function ratio:
$
  Z_1 = sum_(E in bar(L)_1) e^(-beta H(E)), quad Z_2 = sum_(E in bar(L)_2) e^(-beta H(E))
$
If $Z_1 > Z_2$, we infer logical class 1.

Exact computation of $Z$ is #smallcaps("#P-complete"), which is why QEC decoders based on tensor networks or renormalization group aim to approximate this sum efficiently.


== 4. QEC decoding algorithm analysis

In QEC, decoding is the problem of using the syndrome $S$ to recover logical information. For #link(<DEGENERATE_CODES>)[#text(fill:red)[degenerate codes]], this is exactly the **MMAP** problem:
$ hat(L) = "argmax"_L sum_(E in L) P(E | S) $

Here, "sum" handles degeneracy (different physical errors with the same logical effect), and "max" selects the most likely logical class.

=== 4.1 Baseline: #link(<BP_ALGORITHM>)[#text(fill:red)[Belief Propagation (BP)]]

Before evaluating other algorithms, we must clarify BP's role.

==== 4.1.1 Two modes of BP
BP passes messages on a factor graph to update beliefs.
- **Sum-Product BP**: computes marginals $P(x_i | S)$.
  - *In QEC*: computes error probability for each physical qubit (summing over all error combinations, i.e., "Sum").
- **Max-Product BP**: computes the global MAP configuration.
  - *In QEC*: finds the most likely specific error chain (similar to MWPM).

==== 4.1.2 Limitations of BP in QEC
Although BP is $O(N)$ and efficient, it faces three challenges in QEC:
1.  #link(<LOOP_PROBLEM>)[#text(fill:red)[**Loop problem (short cycles)**]]: #link(<QUANTUM_CODES>)[#text(fill:red)[quantum codes]] (especially high-threshold codes) often have many short cycles, causing BP to oscillate or fail to converge.
2.  **MMAP compatibility**: standard BP is either all Sum or all Max. MMAP requires Sum over some variables (physical degeneracy) and Max over others (logical classes). Using Sum-Product BP with hard decisions is a heuristic MMAP approximation.
3.  **Validity failure**: the converged BP result may not satisfy parity checks (i.e., $H hat(e) != S$).


=== 4.2 Deep evaluation of AI-recommended approaches

Given QEC requirements (high throughput, low latency, degeneracy), we analyze AI-proposed solutions one by one.

==== Category 1: Exact inference
*(A1) Junction Tree / (A2) AND/OR Search*

- **Analysis**: These methods decompose by treewidth.
- **QEC applicability**: #text(fill: blue)[*Very low*].
  - High-performance quantum codes (Surface Code, Expander Codes) are designed with high connectivity and treewidth to avoid logical errors.
  - Exact methods are exponential: $O(exp("treewidth"))$.
- **Conclusion**: Only suitable for very small codes (5-qubit, 7-qubit), not practical.

==== Category 2: Approximate with bounds
*(B1) Mini-Bucket / (B2) TRW (Tree-Reweighted) / (B3) LP relaxation*

- **(B1) Mini-Bucket & (B2) TRW**:
  - **Idea**: relax graph structure or use convex optimization to compute upper/lower bounds on $Z$.
  - **Pros**: better convergence than standard BP, less oscillation.
  - **Cons**: higher computational cost and harder to parallelize.
  - **QEC potential**: TRW can outperform BP on some hard codes, but speed is a bottleneck.

- **(B3) LP (linear programming) / ADMM**:
  - **Idea**: model decoding as an LP. MWPM can be seen as a dual LP.
  - **QEC potential**: #text(fill: olive)[*Medium*]. For non-degenerate codes, LP decoders have guarantees, but for degenerate codes (needs Sum), LP often ignores entropy benefits from degeneracy.

==== Category 3: Fast heuristic approximations
*This is the most active area in QEC research, and the direction you should focus on.*

- **(C1) Sum-Max BP (Mixed-Product BP)**:
  - **Idea**: theoretically closest to MMAP. Define two variable classes and use Sum for one and Max for the other.
  - **QEC issue**: numerically unstable. Mixing Sum and Max in floating point often causes underflow or precision loss. Rarely successful in QEC.

- **(C2) BP + Decimation**:
  - **Idea**: run BP $arrow$ fix the most confident bits $arrow$ simplify the graph $arrow$ repeat.
  - **QEC applicability**: #text(fill: green)[*High*]. It breaks symmetry and loops, helping BP converge. A standard enhancement.

- **(C3) GBP (Generalized BP)**:
  - **Idea**: pass messages between regions, not just nodes.
  - **QEC applicability**: #text(fill: olive)[*Very high (for topological codes)*].
  - #link(<SURFACE_CODE>)[#text(fill: red)[Surface Code]] has many short cycles (plaquettes). GBP explicitly handles these loops and significantly improves thresholds. Complexity grows exponentially with region size.

- **(C4) #link(<BP_DETAILED_CODE>)[#text(fill:red)[BP]] + OSD (Ordered Statistics Decoding)**:
  - **Idea**: run Sum-Product BP to get soft information, then apply OSD as post-processing. OSD orders by reliability and flips the least reliable bits to satisfy checks.
  - **QEC applicability**: #text(fill: blue)[*Current SOTA (best practice)*].
  - **Reason**: combines BP speed (soft info) with linear-algebra rigor (guaranteed valid correction). The standard benchmark for #link(<LDPC_CODES>)[#text(fill: red)[LDPC codes]] is BP+OSD.

=== 4.3 Summary and roadmap

If your goal is efficient MMAP decoding, a recommended path is:

#table(
  columns: (1fr, 2fr, 2fr),
  inset: 10pt,
  align: horizon,
  stroke: 0.5pt + gray,
  table.header(
    [*Phase*], [*Algorithm choice*], [*Reason*]
  ),
  [1. Start], 
  [**Sum-Product BP**], 
  [Implement basic message passing as a baseline. Use log-domain computation to avoid underflow.],

  [2. Intermediate (practical)], 
  [**BP + OSD (Order-0 / Order-E)**], 
  [This is the **industry standard**. It fixes BP non-convergence or invalid outputs. Higher OSD order improves accuracy but costs speed.],

  [3. Code-specific], 
  [**GBP (generalized BP)**], 
  [If you mainly study **surface codes** or **color codes** (many short cycles), GBP outperforms BP + OSD.],

  [4. Exploratory], 
  [**Neural BP / Weighted BP**], 
  [Train weights in BP using ML to approximate MMAP.]
)

#block(fill: luma(240), inset: 10pt, radius: 4pt)[
  *Core conclusion*: Do not attempt exact Junction Tree or direct Sum-Max BP. Start with **BP + OSD**. It uses OSD post-processing to compensate for BP's inability to fully solve MMAP, and is the most cost-effective MMAP approximation today.
]

#include"upg_note.typ"
