#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2cm),
  numbering: "I",
)
#set text(
  font: ("Linux Libertine", "Source Han Serif SC"), 
  lang: "en",
  size: 11pt
)
#show math.equation: set text(font: "Latin Modern Math")

// Nice explanation box style
#let note-box(title, color, body) = block(
  fill: color.lighten(90%),
  stroke: (left: 4pt + color),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
  [
    #text(fill: color, weight: "bold", size: 12pt)[#title]
    #v(0.5em)
    #body
  ]
)

#set heading(numbering: none)

= Note

== 1. #link(<NP_PP-complete_return>)[#text(fill:blue)[$("NP")^(PP)$-complete (complexity hierarchy)]] <NP_PP-complete>

This class describes the difficulty of computational problems at a high level of the Polynomial Hierarchy, implying extreme computational hardness.

#note-box("Intuition: nested decision and summation", blue)[
  To understand this, break it into two layers:
  - **NP (Non-deterministic Polynomial)**: represents nondeterministic polynomial time, corresponding to the **MAX operation** in MMAP. This is like searching for an optimal solution in a huge space (e.g., TSP).
  - **PP (Probabilistic Polynomial)**: corresponds to the **SUM operation** in MMAP. It requires summing over all possibilities or marginal probabilities, often harder than just finding an optimum.
  
  **Meaning of $("NP")^(PP)$**:
  This is an "oracle" machine. It means we must solve an NP-hard optimization problem, but to verify each candidate, we must first solve a PP-hard summation problem.
  
  **Conclusion**: This is harder than NP-complete or PP-complete alone. Since exact solutions are infeasible in polynomial time, in QEC or large-scale probabilistic inference we **must** rely on approximate algorithms (e.g., variational inference or dual decomposition).
]

== 2. #link(<Lagrange_Multipliers_return>)[#text(fill:blue)[Lagrange Multipliers]] <Lagrange_Multipliers>

In optimization theory, this is a core technique for constrained problems. In dual decomposition, it acts as a "coordination variable."

#note-box("Mechanism: reach consensus via prices", orange)[
  When we decompose a complex global problem into independent subproblems (e.g., subgraph A and B), these subproblems can disagree on shared variables.
  
  - **Hard constraint**: require $x_A = x_B$. Solving with hard constraints is difficult.
  - **Relaxation**: remove hard constraints and add Lagrange multiplier $delta$ as a penalty in the objective.
  
  **Role of $delta$**:
  Think of $delta$ as the **price of inconsistency**.
  - If subproblem A predicts a higher value than B, the algorithm adjusts $delta$ to "fine" A and "subsidize" B.
  - By iteratively updating $delta$ (usually via subgradient methods), we force each subproblem to approach global consistency while optimizing locally.
]

== 3. #link(<Variational_Upper_Bound_return>)[#text(fill:blue)[Variational Upper Bound]] <Variational_Upper_Bound>

When the objective is intractable (e.g., partition function or marginal likelihood), we build a tractable function that always upper-bounds the true objective.

#note-box("Geometric intuition: lowering the envelope", green)[
  Suppose the true optimum is $Phi^*$ (the true MMAP log-probability). Computing it directly requires high-dimensional sums/integrals.
  
  **Variational strategy:**
  1. **Construct the dual function $L(delta)$**: via dual decomposition. By weak duality, for any $delta$, $L(delta) >= Phi^*$.
  2. **Minimize the upper bound**: since $L(delta)$ is always above $Phi^*$, we search for $delta$ that lowers it.
  3. **Approximation**: as we lower this "ceiling," it approaches the true $Phi^*$.
  
  At convergence, the value may still be approximate, but the **upper bound** provides a theoretical guarantee on solution quality (the duality gap).
]

// (Sections 1-3 above were previously here; omitted to save space, continuing with section 4)

== 4. #link(<Dual_Decomposition_return>)[#text(fill:blue)[Dual Decomposition]] <Dual_Decomposition>

This is the core algorithmic framework for complex graphical model inference. Its philosophy is "divide and coordinate."

#note-box("Core logic: split and negotiate", purple)[
  For a complex global problem (e.g., MMAP on a 2D grid), direct solution is extremely hard due to tight coupling. Dual decomposition proceeds as follows:
  
  1.  **Decompose**: cut some variable interactions, split the big graph into disjoint, easy subgraphs (trees or chains).
  2.  **Solve locally**: each subgraph performs inference independently. Because the structure is simple, this is fast (polynomial time).
  3.  **Coordinate**: the split is artificial, so subgraphs may disagree on shared variables. We introduce **dual variables** (see section 2) to penalize disagreement and drive consensus.
  
  This is like distributing a big project to multiple teams; the manager (master algorithm) adjusts incentives so outputs align.
]

== 5. #link(<Grid_Decomposition_return>)[#text(fill:blue)[Grid decomposition details (Row/Col Decomposition)]] <Grid_Decomposition>

For surface codes or 2D Ising models, how do we decompose an $N times N$ grid into easy structures?

#note-box("Operational details: edge-based split", red)[
  A 2D grid has **nodes** and **edges**. The difficulty comes from **loops**. Our goal is to remove loops while keeping tree structures.
  
  **Steps:**
  1.  **Duplicate nodes**: for each node $x_(i,j)$, create a copy $x_(i,j)^("row")$ in the horizontal set and $x_(i,j)^("col")$ in the vertical set.
  2.  **Assign edges (key step)**:
      - **Row strips**: keep only **horizontal edges**. The grid becomes $N$ independent horizontal chains (1D, no loops).
      - **Col strips**: keep only **vertical edges**. The grid becomes $N$ independent vertical chains.
  3.  **Result**: instead of one complex 2D grid, we now have $2N$ simple 1D chains.
  
  This decomposition is ideal for parallel computation because each chain's inference (transfer matrix or forward-backward) is independent.
]

== 6. #link(<Consensus_Constraint_return>)[#text(fill:blue)[Consistency constraint (Why Consistency?)]] <Consensus_Constraint>

In step 5, we created "row copies" and "col copies." Why must we enforce their agreement?

#note-box("Logic: returning to physical reality", teal)[
  **Why consistency?**
  In the real physical system (original problem), qubit $(i,j)$ is unique.
  - It cannot be "error" ($x=1$) from the row view while "no error" ($x=0$) from the column view.
  - If copies disagree, the solution violates physical reality and is invalid.
  
  **Role of Lagrangian relaxation:**
  - Ideally, we want a hard constraint: $x^("row") = x^("col")$. This is hard to enforce directly.
  - **Relaxation**: allow temporary disagreement, but penalize it with Lagrange multipliers $delta$.
  - At convergence, if penalties work, the copies converge to the same value, and $L(delta)$ equals (or closely approximates) the original optimum.
]
