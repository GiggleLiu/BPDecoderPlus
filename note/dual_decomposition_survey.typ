#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2cm),
  numbering: "1",
)
#set text(
  font: ("Linux Libertine", "Source Han Serif SC"), 
  lang: "en",
  size: 11pt
)
// Enable math font support
#show math.equation: set text(font: "Latin Modern Math")

// Heading style
#set heading(numbering: "1.1.")
#show heading: it => [
  #v(0.5em)
  #block(it)
  #v(0.3em)
]

// Emphasis box style (same as original)
#let term(name, body) = block(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [**#name**: #body]
)

// Custom operators in math blocks
#let argmax = math.op("argmax", limits: true)
#let argmin = math.op("argmin", limits: true)
#let Inference = math.op("Inference")

= MMAP via Dual Decomposition: A Detailed Survey

== Background and problem definition

**MMAP (Marginal Maximum A Posteriori)** is a challenging inference task in probabilistic graphical models. It combines summation (marginalization) and maximization (MAP).

Given a set of random variables $X = X_M union X_S$, where $X_M$ are the variables to optimize (Max variables) and $X_S$ are the variables to sum out (Sum variables).

The joint distribution is typically defined in exponential-family form:
$P(x) ∝ exp(sum_(α in cal(I)) θ_α(x_α))$

where $theta_alpha$ is a potential function defined over subset $x_alpha$.

The MMAP objective is:
$ x_M^* = argmax_(x_M) sum_(x_S) exp(sum_alpha theta_alpha (x_alpha)) $

Equivalently, maximize the log-probability:
$ Phi^* = max_(x_M) log sum_(x_S) exp(sum_alpha theta_alpha (x_alpha)) $

Because of coupling between $X_M$ and $X_S$, this problem is usually #link(<NP_PP-complete>)[#text(fill:red)[$"NP"^(PP)$-complete]], harder than standard MAP inference. <NP_PP-complete_return>

== Core idea of Dual Decomposition

The core idea is to decompose a hard global problem into easier local subproblems (slave problems), and enforce consistency via #link(<Lagrange_Multipliers>)[#text(fill:red)[Lagrange multipliers]]. <Lagrange_Multipliers_return>

In MMAP, we often use a #link(<Variational_Upper_Bound>)[#text(fill:red)[variational upper bound]] to construct the dual problem. <Variational_Upper_Bound_return>

=== Problem reformulation and variable copies

Assume we decompose the original graph into a set of subgraphs (e.g., trees) $cal(T)$. For each subgraph $t in cal(T)$, we introduce local variable copies $x^t$ and local potentials $theta^t$.

To ensure equivalence to the original problem, we must satisfy:
1. **Potential conservation**: $sum_t theta_alpha^t = theta_alpha$ (often evenly split).
2. **Variable consistency**: for all overlapping variables $i$, copies must agree across subgraphs: $x_i^t = x_i^(t')$.

=== Lagrangian relaxation

We incorporate the consistency constraints $x_i^t = x_i$ into the objective. For MMAP, we must enforce both Max-variable consistency and, in the variational view, marginal consistency for Sum variables.

Construct the dual objective $L(delta)$, where $delta$ are dual variables (Lagrange multipliers):

$ L(delta) = sum_(t in cal(T)) Phi^t (theta^t + delta^t) $

where:
- $Phi^t$ is the local MMAP objective on subgraph $t$ (log-sum-exp form).
- $delta^t$ are adjustment terms assigned to subgraph $t$, satisfying $sum_t delta^t = 0$.

By weak duality, for any feasible $delta$:
$ L(delta) >= Phi^* $
so $L(delta)$ is an upper bound on the primal optimum.

== Algorithm: projected subgradient method

We minimize the dual function to tighten the bound:
$ min_delta L(delta) quad text("s.t.") quad sum_t delta^t = 0 $

We typically use **projected subgradient descent**.

=== Steps

1. **Initialize**: set $delta^t = 0$.
2. **Iterate** (until convergence):
   + **Solve subproblems (Slave step)**:
     For each subgraph $t$, independently solve the local MMAP problem. Compute local marginals for Sum variables or optimal assignments for Max variables.
     We compute the "pseudo-marginals" (beliefs) $mu_i^t (x_i)$ for variables in subgraph $t$.
     
     $ mu_i^t (x_i) arrow.l Inference(theta^t + delta^t) $
     
     _Note: if the subgraph is a tree, this can be done efficiently with sum-product or max-sum variants in polynomial time._

   + **Subgradient computation**:
     The subgradient reflects disagreement between beliefs across subgraphs.
     $ g_i^t (x_i) = mu_i^t (x_i) - 1 / abs(cal(T)_i) sum_(k in cal(T)_i) mu_i^k (x_i) $
     where $cal(T)_i$ is the set of subgraphs containing variable $i$.

   + **Dual update**:
     $ delta_i^t (x_i) arrow.l delta_i^t (x_i) - eta dot g_i^t (x_i) $
     where $eta$ is the step size schedule.

3. **Decoding**:
   If at some iteration all subgraphs agree on the optimal assignments of Max variables $X_M$, we have an exact solution. Otherwise, decode heuristically from the beliefs.

== Key details

=== Difficulty of subproblems
Efficiency depends on whether $Phi^t$ is easy to solve.
- If subgraphs are **trees**: local MMAP can be solved exactly via mixed-product message passing.
- If subgraphs are complex: additional approximations or simpler structures (e.g., edge decomposition) may be needed.

=== Why it works
1. **Upper-bound guarantee**: always provides an upper bound on the objective, useful for assessing solution quality (duality gap).
2. **Parallelism**: each subgraph inference is independent, easy to parallelize.
3. **Flexibility**: decomposition choices (star, tree, etc.) are flexible.

== Summary

MMAP via #link(<Dual_Decomposition>)[#text(fill:red)[dual decomposition]] proceeds as follows: <Dual_Decomposition_return>
1. **Relax coupling**: break complex dependencies and introduce copies.
2. **Local solve**: exploit tree structure for efficient local MMAP.
3. **Global coordination**: penalize inconsistency with Lagrange multipliers to approach the global optimum.

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)
#v(0.5em)
_Note: the formulas above follow a generic variational dual derivation; implementation details can vary by decomposition strategy (e.g., TRW, DD-ADMM)._

= Dual Decomposition for QEC: Detailed Application

== DD-based QEC decoding

Directly computing MMAP summation is hard: $ L^* = argmax_(L) P(L | S) = argmax_(L) sum_(E in L) P(E | S) $. We use dual decomposition to split a complex 2D grid into simple 1D chain problems.

We use the **Surface Code** as the canonical example.

=== Step 1: Build the factor graph
The surface code is a 2D grid. We model it as a probabilistic graphical model (often an Ising model):
- **Nodes**: error states on physical qubits (error=1, no error=0).
- **Factors**: constraints. If two errors are adjacent, they must satisfy syndrome parity constraints.

The joint distribution is:
$ P(E) ∝ exp(beta sum_((i,j)) J_(i j) x_i x_j) $
where $beta$ relates to the physical error rate and $J_(i j)$ encodes syndrome information.

=== Step 2: Decomposition
Inference on the 2D grid is hard. If we cut it into horizontal **chains** or **trees**, inference becomes easy on each chain.

#link(<Grid_Decomposition>)[#text(fill:red)[We decompose the original $N times N$ grid into two sets:]]<Grid_Decomposition_return>
1. **Horizontal strip set**: contains all horizontal connections.
2. **Vertical strip set**: contains all vertical connections.

Each qubit $x_i$ now has two copies: $x_i^("row")$ and $x_i^("col")$.

=== Step 3: Lagrangian relaxation (dual objective)
#link(<Consensus_Constraint>)[#text(fill:red)[We want these two copies to agree: $x_i^("row") = x_i^("col")$.]] We introduce Lagrange multipliers $delta_i$ to penalize inconsistency. <Consensus_Constraint_return>
We introduce Lagrange multipliers $delta_i$ to penalize inconsistency.

The dual objective becomes:
$ L(delta) = sum_(t in "Strips") Phi^t (theta^t + delta^t) $
where $Phi^t$ is the local MMAP energy on strip $t$.

=== Step 4: Iterative optimization (the loop)

This is a repeated negotiation process.

**1. Local subproblem solve (Slave step)**
For each strip (chain), compute "what is the probability that qubit $i$ is in error from this strip's view?"
Since each strip is 1D, we can efficiently use the **transfer matrix** or **forward-backward** algorithm.

Computation:
- Input: local potential $theta^t$ plus dual variable $delta^t$.
- Operation: run Sum-Product (since we sum over physical errors).
- Output: **pseudo-marginals (beliefs)** $mu_i^t(x_i)$.


#block(fill: luma(240), inset: 8pt, radius: 4pt)[
  On a 1D chain, we can define a matrix $M$. The chain partition function is a product: $Z = v_L^T M_1 M_2 ... M_N v_R$.
  This step is linear time $O(N)$, very fast.
]

**2. Coordination and update (Master step)**
Compare the horizontal and vertical strip beliefs.
- If the row strip believes $x_i$ is likely wrong ($mu > 0.5$), while the column strip believes no error ($mu < 0.1$), there is disagreement.
- The gradient $g_i$ is this disagreement:
  $ g_i = mu_i^("row") - mu_i^("col") $
- Update dual variables $delta$:
  $ delta_i arrow.l delta_i - eta dot g_i $
  
  _Intuition: if the row strip is overconfident, increase its penalty to reduce confidence next round, and vice versa._

=== Step 5: Logical-class decision (Decoding)
When the algorithm converges (or hits max steps), we do not require exact agreement at the bit level. We need a decision over the **logical class $L$**.

We can estimate the approximate partition function $Z(delta)$ for each logical class ($I, X, Y, Z$) and choose the largest.

$ "Result" = argmax_(L in {I, X, Y, Z}) Z_L $

== Algorithm flow summary

For clarity, the process is summarized as:

+ **Input**: syndrome $S$ (check outcomes).
+ **Initialize**: $delta = 0$.
+ **Outer loop (try logical classes $L$)**:
  Assume the logical error is $L$ (e.g., logical flip), add a global constraint.
  
  + **Inner loop (DD optimization)**:
    1. **Parallel compute**: all rows and columns run Sum-Product independently.
    2. **Compute beliefs**: get marginals for each qubit.
    3. **Update $delta$**: adjust weights based on row/column disagreement.
  + **Output**: approximate probability (free energy) for that $L$.
+ **Final decision**: compare $L=I$ and $L=X$, choose the larger.

== Why do this? (Pros and cons)

**Pros:**
1. **Accounts for degeneracy**: key difference from MWPM (minimum weight perfect matching). MWPM finds the shortest path (MAP), while DD-MMAP sums over all paths. At higher noise, DD-MMAP yields better accuracy and threshold.
2. **Parallelism**: each strip is independent, well-suited for GPU/FPGA acceleration.

**Challenges:**
1. **Convergence**: dual decomposition does not always converge to the optimum; it can oscillate near the optimum.
2. **Compute cost**: faster than brute-force tensor contraction, but still slower than MWPM.


= Algorithm comparison: DD-MMAP vs. BP+OSD

In QEC decoding (especially qLDPC and surface codes), **DD (Dual Decomposition)** and **BP+OSD (Belief Propagation + Ordered Statistics Decoding)** are two representative variational/approximate inference methods.

Both leverage factor graphs, but their philosophies and strategies for handling loops differ greatly.

== 1. Core mechanism differences

=== BP+OSD: heuristic iterations + linear-algebra repair
BP+OSD is the current "gold standard" for qLDPC decoding. It has two stages:
1.  **BP stage (Belief Propagation)**:
    Run sum-product or min-sum directly on the loopy factor graph.
    - _Issue_: short loops (especially in surface codes) break independence assumptions, causing oscillations or convergence to wrong pseudocodewords.
2.  **OSD stage (Ordered Statistics Decoding)**:
    A post-processing step. It uses BP's soft outputs (LLRs) to correct the least reliable bits.
    - It sorts by reliability, selects a linearly independent basis, and uses Gaussian elimination to force a syndrome-satisfying solution.

=== DD-MMAP: structured decomposition + optimization guarantees
DD does not force computation on a loopy graph; it changes the structure:
1.  **Decomposition**:
    Explicitly cut loopy graphs into acyclic subgraphs (trees or chains).
2.  **Exact inference + coordination**:
    Perform **exact** inference inside each subgraph (because it is acyclic), then coordinate them with Lagrange multipliers.

#term("One-sentence summary")[
  - **BP+OSD**: guess first (BP), then patch with linear algebra (OSD).
  - **DD-MMAP**: split hard problems into easy ones, then negotiate ($delta$ updates) to agree.
]

== 2. Handling degeneracy

This is a key difference in QEC decoding.

- **BP+OSD limitations**:
  BP tries to compute marginals (sum), but loopy graphs make results inaccurate. OSD effectively seeks **one** best physical error (MAP), not the sum over a logical class. Thus BP+OSD often underperforms on highly degenerate codes (like surface codes) compared to sparse LDPC codes.

- **DD-MMAP advantages**:
  DD's objective directly models $sum exp(...)$. In the slave step, it explicitly runs sum-product on subgraphs, so it **naturally sums over all error paths**. Theoretically, DD aligns more closely with the MMAP goal (optimal logical class rather than optimal physical error).

== 3. Convergence and theoretical bounds

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 6pt,
  stroke: 0.6pt + luma(70%),

  table.header(
    [*Property*], [*BP+OSD*], [*DD-MMAP*],
  ),

  [*Objective*],
  [No explicit global objective (BP is local message passing)],
  [Variational dual objective $L(δ)$],

  [*Guarantees*],
  [None. BP may fail to converge.],
  [*Strong*. Provides an upper bound on the log-partition function.],

  [*Convergence behavior*],
  [Fast, usually tens of iterations.],
  [Slower; subgradient methods need many iterations and may oscillate near the optimum.],
)


== 4. Complexity and speed

- **BP+OSD (faster)**:
  - BP complexity is $O(N)$ (linear in edges).
  - OSD cost is mainly Gaussian elimination, worst-case $O(N^3)$, but typically fast for sparse matrices.
  - **In practice**: very fast, suitable for real-time decoding.

- **DD-MMAP (slower)**:
  - Each iteration runs forward-backward on all subgraphs ($O(N)$).
  - Typically requires many more iterations than BP (hundreds or thousands for high precision).
  - **In practice**: used for offline evaluation, theoretical bounds, or high-accuracy benchmarks; less real-time than BP+OSD.

== 5. Summary table

#figure(
  table(
    columns: (1.5fr, 2fr, 2fr),
    inset: 10pt,
    align: horizon,
    stroke: 0.5pt + gray,
    fill: (col, row) => if row == 0 { luma(230) } else { white },
    
    [*Dimension*], [*BP + OSD*], [*Dual Decomposition (DD)*],
    
    [**Core idea**], 
    [Heuristic message passing + linear-system postprocessing], 
    [Lagrangian relaxation + exact inference on subgraphs],
    
    [**Loop handling**], 
    [Ignore loops; correct with OSD], 
    [Physically cut loops; penalize inconsistency],
    
    [**Degeneracy support**], 
    [Weak (tends to find a single error pattern)], 
    [Strong (naturally sums over all paths)],
    
    [**Main advantage**], 
    [Fast and mature; good for LDPC], 
    [Theoretical upper bound; high accuracy on surface codes],
    
    [**Main drawback**], 
    [Performance drops on short-loop graphs; no convergence guarantee], 
    [Many iterations; high compute cost; not real-time],
  ),
  caption: [Detailed comparison of BP+OSD and DD-MMAP]
)

== 6. Why both matter in QEC

Although BP+OSD dominates practical deployments (e.g., Google, IBM), DD-MMAP provides unique value:

1.  **Benchmarking**: DD gives an upper bound to assess how far BP+OSD is from "perfect" (duality gap).
2.  **Hybrid strategies**: recent work combines both, e.g., using DD decomposition to initialize BP, or BP outputs to guide DD subgraphs, aiming to combine speed and accuracy.

#include"dd_note_trans.typ"
