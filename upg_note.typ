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
  leading: 0.8em
)

// Define a nice note block style
#let note(title, body) = {
  block(
    fill: luma(245),
    stroke: (left: 4pt + blue.darken(20%)),
    inset: 12pt,
    radius: (right: 4pt),
    width: 100%,
    [
      #text(weight: "bold", fill: blue.darken(20%), size: 1.1em)[#title]
      #v(0.5em)
      #body
    ]
  )
}

= Note

== 1. What is a maximal clique? <MAXIMAL_CLIQUE>

In an undirected graph, the concept of a clique is the basis for defining local relationships.

#note("Definitions and Distinctions")[
  1. *Clique*:
     A subset of nodes such that every pair of nodes in the subset is connected by an edge (a fully connected subgraph).

  2. *Maximal clique*:
     A clique that cannot be expanded by adding any other node while remaining a clique.

  *Example*:
  Suppose an undirected graph $G$ has node connections $A-B, B-C, A-C, B-D$.
  - ${A, B}$ is a clique.
  - ${A, B, C}$ is a clique (a triangle), and you cannot add $D$ (since neither $A$ nor $C$ connects to $D$), so ${A, B, C}$ is a *maximal clique*.
  - ${B, D}$ is also a *maximal clique*.
]

*Role*: In a Markov random field, maximal cliques define the smallest units for factorizing the probability distribution. If two variables are not in the same maximal clique, it means they have no direct strong coupling (or their direct relation is already contained in a larger clique).

#v(1em)

== 2. What is a potential function? <POTENTIAL_FUNCTION>

Because an undirected graph has no direction, we cannot define conditional probabilities like $P(A|B)$ (since $A$ and $B$ are on equal footing). Instead, we use a potential function to quantify compatibility between variables.

#note("Properties and Meaning")[
  *Definition*:
  A non-negative real-valued function $psi_C(X_C)$ defined on a maximal clique $C$, where $X_C$ is the set of variables in the clique.

  *Key features*:
  1. *Non-negativity*: $psi_C(X_C) >= 0$.
  2. *Not a probability*: A potential function is not a probability and does not need to be normalized (the sum does not have to be 1).
  3. *Intuition*:
     - A larger $psi$ value indicates a more likely configuration.
     - It can be seen as the inverse of energy. Often $psi_C(x) = exp(-E(x))$, where $E(x)$ is an energy function. Lower energy implies a more stable system and a higher probability.
]

*Example*: In image processing, adjacent pixels $x_i, x_j$ tend to have similar colors. We can define a potential $psi(x_i, x_j)$ that takes a large value when $x_i approx x_j$, and a small value when they differ.

#v(1em)

== 3. Why can the joint probability factor into a product of potentials? <HAMMERSLEY_CLIFFORD_THEOREM>

This is one of the deepest results in graphical models, known as the *Hammersley-Clifford theorem*.

#note("Hammersley-Clifford Theorem")[
  *Question*:
  We want the joint distribution $P(X)$ to satisfy the conditional independence implied by the graph structure (Markov property).

  *Theorem*:
  If a distribution $P(X) > 0$ (strictly positive) satisfies the local Markov property defined by an undirected graph $G$, then it can be factorized as a product of potential functions over all maximal cliques in the graph:

  $
    P(X) = 1/Z product_(C in cal(C)) psi_C(X_C)
  $
]

*Why is this true? (Intuition)*

1. *Localization principle*:
   Graph theory tells us that direct interactions are limited to nodes connected by an edge. Maximal cliques contain all groups of variables with direct mutual influence.

2. *Independence is reflected*:
   If two variables $x_i$ and $x_j$ never appear in the same potential $psi_C$, it means there is no direct interaction between them. This matches the absence of an edge in the graph and ensures conditional independence (given a separating set).

3. *Necessity of a product form*:
   When we consider two independent subsystems (a disconnected graph), the joint probability should factor as $P(A, B) = P(A)P(B)$. A product of potentials naturally satisfies this, while a sum does not.


== 4. Partition function $Z$ in detail: what are "all possible variable configurations"? <PARTITION_FUNCTION>

In the definition of a Markov random field, the partition function $Z$ is written as:
$ Z = sum_X product_(C in cal(C)) psi_C(X_C) $

The term $sum_X$ is often the hardest part for beginners. It is not a sum over numeric values, but an enumeration over **all possible world states**.

#note("Definition of a Configuration")[
  A "configuration" (assignment) means assigning a specific value to **every** random variable in the graph at the same time.

  If the graph has $n$ variables $X_1, X_2, dots, X_n$, then "all possible configurations" are the **Cartesian product** of their values.
]

=== 4.1 Example: understanding $Z$ with 3 variables

Suppose we have a very simple model with only 3 variables $A, B, C$.
- Assume they are all binary (e.g., heads/tails), with values ${0, 1}$.
- Assume the model defines a global potential $psi(A, B, C)$ (for simplicity, no clique decomposition).

Then there are $2^3 = 8$ possible configurations. Computing $Z$ means summing the "unnormalized probabilities" from these 8 parallel worlds.

We list a "truth table" to show how $Z$ is computed:


#figure(
  table(
    columns: (1fr, 1fr, 1fr, 3fr),
    inset: 8pt,
    align: horizon + center,
    stroke: 0.5pt + luma(200),
    table.header(
      [*Variable $A$*], [*Variable $B$*], [*Variable $C$*], [*"Score" of this configuration* \\ $s_i = psi(A,B,C)$]
    ),
    [0], [0], [0], [$s_1 = psi(0,0,0)$],
    [0], [0], [1], [$s_2 = psi(0,0,1)$],
    [0], [1], [0], [$s_3 = psi(0,1,0)$],
    [0], [1], [1], [$s_4 = psi(0,1,1)$],
    [1], [0], [0], [$s_5 = psi(1,0,0)$],
    [1], [0], [1], [$s_6 = psi(1,0,1)$],
    [1], [1], [0], [$s_7 = psi(1,1,0)$],
    [1], [1], [1], [$s_8 = psi(1,1,1)$],
    table.cell(colspan: 3, align: right, text(weight: "bold")[Partition function $Z =$]),
    table.cell(fill: yellow.lighten(90%), [$s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7 + s_8$])
  ),
  caption: [Computing $Z$ requires traversing all rows]
)

After computing $Z$, if we want the probability of a specific state (for example all 1s), we divide that row's score by the total:
$ P(A=1, B=1, C=1) = s_8 / Z $

=== 4.2 Why is this hard? (The Partition Function Problem)

#note("Exponential blowup")[
  In the example above, 3 variables require 8 sums. It seems easy for a computer.

  But in real applications (e.g., image processing):
  - A small image might have $100 times 100$ pixels.
  - The number of variables is $n = 10,000$.
  - Each pixel has 2 values (black/white).
  - Total configurations = $2^(10,000)$.

  $2^(10,000) approx 10^(3000)$.

  For comparison, the total number of atoms in the observable universe is about $10^80$. This means no computer can compute $Z$ exactly by "traversing all configurations".
]

That is why in undirected models (such as CRFs and RBMs) we usually cannot do exact inference, and instead use:
1.  **Sampling**: e.g., MCMC, which "walks" around high-probability configurations without traversing all of them.
2.  **Approximate inference**: e.g., variational inference, which fits a simple distribution to approximate a complex one.

// ==========================================
// Drawing helper functions (place at the beginning or before the current section)
// ==========================================

#let draw-node(x, y, label, is-factor: false) = {
  place(
    top + left,
    dx: x, dy: y,
    if is-factor {
      rect(width: 20pt, height: 20pt, fill: black, radius: 2pt)[
        #align(center + horizon, text(fill: white, size: 8pt, label))
      ]
    } else {
      circle(radius: 12pt, stroke: 1pt + black, fill: white)[
        #align(center + horizon, text(size: 10pt, label))
      ]
    }
  )
}

#let draw-edge(x1, y1, x2, y2) = {
  place(top + left, line(start: (x1, y1), end: (x2, y2), stroke: 1pt + gray))
}

= 5. Relationship between Tanner graphs, factor graphs, and undirected graphs <GRAPH_RELATIONSHIP>

In QEC and probabilistic inference, these terms are often used interchangeably, but they live at different levels of abstraction. We can see them as a relationship of **containment and concretization**.



== 5.1 Conceptual hierarchy

In QEC and probabilistic inference, these three concepts represent different levels from an "abstract mathematical model" to a "concrete implementation structure". We compare them visually.

We use a simple **three-variable correlation** model as an example: variables $A, B, C$, as shown below.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 10pt,
    
    // --- Graph 1: Undirected probabilistic graph (MRF) ---
    block(height: 130pt, width: 100%, stroke: 0.5pt + gray, radius: 5pt, inset: 10pt)[
      #align(center, [*1. Undirected probabilistic graph (MRF)*])
      #v(10pt)
      // Coordinate definitions
      #let (ax, ay) = (20pt, 60pt)
      #let (bx, by) = (80pt, 60pt)
      #let (cx, cy) = (50pt, 10pt)
      
      // Draw edges (pairwise connections)
      #draw-edge(ax+12pt, ay+12pt, bx+12pt, by+12pt)
      #draw-edge(ax+12pt, ay+12pt, cx+12pt, cy+12pt)
      #draw-edge(bx+12pt, by+12pt, cx+12pt, cy+12pt)
      
      // Draw nodes
      #draw-node(ax, ay, "A")
      #draw-node(bx, by, "B")
      #draw-node(cx, cy, "C")
      
      #place(bottom + center, text(size: 8pt, fill: gray)[Meaning: A,B,C are mutually correlated\ (maximal clique structure)])
    ],

    // --- Graph 2: Factor graph ---
    block(height: 130pt, width: 100%, stroke: 0.5pt + gray, radius: 5pt, inset: 10pt)[
      #align(center, [*2. Factor graph*])
      #v(10pt)
      #let (ax, ay) = (20pt, 60pt)
      #let (bx, by) = (80pt, 60pt)
      #let (cx, cy) = (50pt, 10pt)
      #let (fx, fy) = (50pt, 40pt) // Factor node position
      
      // Draw edges (connected to factor)
      #draw-edge(ax+12pt, ay+12pt, fx+10pt, fy+10pt)
      #draw-edge(bx+12pt, by+12pt, fx+10pt, fy+10pt)
      #draw-edge(cx+12pt, cy+12pt, fx+10pt, fy+10pt)
      
      // Draw nodes
      #draw-node(ax, ay, "A")
      #draw-node(bx, by, "B")
      #draw-node(cx, cy, "C")
      #draw-node(fx, fy, "f", is-factor: true)
      
      #place(bottom + center, text(size: 8pt, fill: gray)[Meaning: explicit function f(A,B,C)\ defines the correlation])
    ],

    // --- Graph 3: Tanner graph ---
    block(height: 130pt, width: 100%, stroke: 0.5pt + gray, radius: 5pt, inset: 10pt)[
      #align(center, [*3. Tanner graph*])
      #v(10pt)
      // Variable nodes (Data Qubits)
      #let (d1x, d1y) = (15pt, 10pt)
      #let (d2x, d2y) = (50pt, 10pt)
      #let (d3x, d3y) = (85pt, 10pt)
      // Check nodes (Check Operators)
      #let (c1x, c1y) = (30pt, 60pt)
      #let (c2x, c2y) = (70pt, 60pt)
      
      // Draw edges (bipartite structure)
      #draw-edge(d1x+12pt, d1y+12pt, c1x+10pt, c1y+10pt)
      #draw-edge(d2x+12pt, d2y+12pt, c1x+10pt, c1y+10pt)
      #draw-edge(d2x+12pt, d2y+12pt, c2x+10pt, c2y+10pt)
      #draw-edge(d3x+12pt, d3y+12pt, c2x+10pt, c2y+10pt)
      
      // Draw nodes
      #draw-node(d1x, d1y, "d1")
      #draw-node(d2x, d2y, "d2")
      #draw-node(d3x, d3y, "d3")
      #draw-node(c1x, c1y, "S1", is-factor: true)
      #draw-node(c2x, c2y, "S2", is-factor: true)

      #place(bottom + center, text(size: 8pt, fill: gray)[Meaning: bipartite structure\ S1 checks d1,d2])
    ]
  ),
  caption: [Structural comparison of three graph models]
)

A simple containment relation is:
$ "Tanner Graph" subset "Factor Graph" subset "Representation of MRF" $

#note("Core differences among the three")[
  1.  **Undirected probabilistic graph (MRF)**:
      This is the **mathematical model**. It abstractly describes correlations between variables (via maximal cliques), but does not necessarily specify the exact factor structure.
  
  2.  **Factor graph**:
      This is a **fine-grained representation** of an MRF. It introduces extra "factor nodes" to explicitly show the scope of potential functions, resolving ambiguity in complex clique structures.
  
  3.  **Tanner graph**:
      This is a **special case** of a factor graph in **coding theory**. It is used to describe linear block codes (such as LDPC codes or stabilizer codes in QEC).
]

== 5.2 Deep dive

=== 5.2.1 From an undirected graph (MRF) to a factor graph

In a standard undirected graph, if three variables $A, B, C$ form a triangle (a clique), we only know they are related, but not whether the relation is pairwise or joint.

*A factor graph* removes this ambiguity by turning the graph into a **bipartite graph**:
- **Circle nodes**: variables $X_i$.
- **Square nodes**: factors $f_j$.
- **Edges**: a variable $X_i$ connects to a factor $f_j$ only if $X_i$ is an argument of $f_j$.

Formula correspondence:
$ P(X) = 1/Z product_j f_j(X_"scope"(j)) $

=== 5.2.2 From a factor graph to a Tanner graph

A Tanner graph is essentially a **factor graph with binary variables**, but with specific physical/logical meanings:

- **Variable nodes**: codeword bits (Data Qubits).
- **Factor nodes**: check bits (Check Qubits / Stabilizers / Parity Checks).
- **Mathematical form of factors**:
  In a Tanner graph, the "potential" is usually a hard constraint or indicator function.
  
  For example, for a parity check, the factor $f$ is defined as:
  $
    f(x_1, x_2, x_3) = cases(
      1\, & "if" x_1 + x_2 + x_3 = 0 space ("mod" 2),
      0\, & "otherwise"
    )
  $
  
  In QEC, this means if the physical error $E$ commutes with the stabilizer $S$, the probability (weight) is non-zero; otherwise it is forbidden (or extremely small in soft decoding).

== 5.3 Summary table

#table(
  columns: (1fr, 2fr, 2fr),
  inset: 10pt,
  align: horizon,
  stroke: 0.5pt + gray,
  table.header(
    [*Name*], [*Structural features*], [*QEC correspondence*]
  ),
  [Undirected probabilistic graph\ (MRF)], 
  [Ordinary graph, nodes are variables and edges are dependencies], 
  [Describes entanglement/statistical correlations between qubits],

  [Factor graph], 
  [Bipartite graph, explicitly separates "variables" and "functions"], 
  [A general decoding framework, the carrier for Belief Propagation (BP)],

  [Tanner graph], 
  [Subset of factor graphs, with factors as parity checks], 
  [Describes stabilizer code structure: variables = physical qubits, factors = stabilizer measurements]
)


// Keep the previous note function definition (if appending in the same file, no need to repeat; if new file, keep it)
#let note(title, body) = {
  block(
    fill: luma(245),
    stroke: (left: 4pt + blue.darken(20%)),
    inset: 12pt,
    radius: (right: 4pt),
    width: 100%,
    [
      #text(weight: "bold", fill: blue.darken(20%), size: 1.1em)[#title]
      #v(0.5em)
      #body
    ]
  )
}

== 6. Differences between MAP and MMAP: from intuition to math <MAP>

In probabilistic inference, MAP and MMAP differ by only one letter, but their logic for finding the "best answer" is completely different. For QEC, understanding this is crucial.

=== 6.1 Definition comparison

Suppose we have two variables:
- $X$: the variable we **care** about (e.g., logical error type).
- $Y$: the variable we **do not care** about but that does exist (e.g., physical error details).
- $E$: observed evidence (e.g., syndrome).

#note("Mathematical definitions")[
  1.  **MAP (Maximum A Posteriori)**
      Find the most likely **specific global configuration** $(x, y)$.
      $
        (x^*, y^*) = "argmax"_(x, y) P(x, y | E)
      $
      "Find the single most likely microscopic scenario."

  2.  **MMAP (Marginal MAP)**
      Find the most likely **target variable configuration** $x$, marginalizing over $y$.
      $
        x^* = "argmax"_x sum_y P(x, y | E) = "argmax"_x P(x | E)
      $
      "First add up all possibilities that share the same $x$, then see which group has the largest total probability."
]

=== 6.2 Intuitive example: election and votes

To see why MAP and MMAP give different answers, consider an **"election paradox"** example.

Suppose a class elects a leader. Candidates belong to two groups: **Logical party (L)** and **Physical party (P)**.
- The Logical party has one candidate: $L_1$.
- The Physical party has three candidates: $P_1, P_2, P_3$.

Vote shares (posterior probabilities) are:

#figure(
  table(
    columns: (1fr, 1fr, 1fr),
    inset: 10pt,
    align: center + horizon,
    stroke: 0.5pt + gray,
    table.header([*Group (macro)*], [*Candidate (micro)*], [*Vote share (probability)*]),
    [*Logical party*], [$L_1$], text(fill: red, weight: "bold")[40%],
    table.cell(rowspan: 3, align: center + horizon)[*Physical party*], [$P_1$], [25%],
    [$P_2$], [20%],
    [$P_3$], [15%],
  ),
  caption: [Decision difference between MAP and MMAP]
)

*Inference comparison*:

1.  **MAP view (highest single point)**:
    - Who is the highest-vote **individual**?
    - Answer: $L_1$ (40%).
    - *Conclusion*: MAP says the Logical party wins.

2.  **MMAP view (group total)**:
    - Which **group** has the highest total vote share?
    - Logical party total: $40%$.
    - Physical party total: $25% + 20% + 15% = 60%$.
    - *Conclusion*: MMAP says the Physical party wins.

#note("Key insight")[
  MAP is easily attracted to a **"spike"** (a single high-probability state).
  
  MMAP focuses on **probability mass**: even if each state is not large, if there are many such states (high degeneracy), their total probability can win.
]

=== 6.3 Meaning in QEC

This explains why MMAP is the correct answer in quantum error correction, while MAP (often approximated by MWPM) is only an approximation.

- **Physical errors (micro-state)**: there can be thousands of concrete error paths (e.g., a chain going slightly left or right), corresponding to different $Y$.
- **Logical errors (macro-state)**: all these paths lead to the same logical operation (e.g., logical $X$ flip), corresponding to $X$.

*QEC status*:
- **MWPM (minimum weight perfect matching)**: effectively does MAP, finding the single most probable error chain. When the error distribution is sharp (low-temperature limit), MAP and MMAP are close.
- **Tensor Network / BP decoders**: attempt to compute MMAP by summing all equivalent error chains (entropy effect). At higher noise, MMAP has a significantly higher decoding threshold than MAP.

// Additional content starts

== 7. What are degenerate codes? <DEGENERATE_CODES>

In classical error-correcting codes, each error usually corresponds to a unique syndrome, so the decoder only needs to find that unique error. In quantum error correction, things get more interesting and complex.

#note("Definition")[
  **Degeneracy** means multiple **different** physical error patterns $E_1, E_2$ have **exactly the same** logical effect on the quantum state (or are equivalent).

  Mathematically, if $E_2 = E_1 dot S$, where $S$ is an element of the stabilizer group, then $E_1$ and $E_2$ are degenerate. The stabilizer acts as the identity $I$ on the code space.
]

*Why is it important?*
- **Bad**: It makes MAP (maximum a posteriori) invalid. MAP tries to distinguish $E_1$ and $E_2$, which is meaningless and wastes computation.
- **Good**: It increases the probability of successful correction. We need **MMAP**, i.e., sum the probabilities of all equivalent errors $E_1, E_2, dots$. This "entropy gain" makes quantum codes more robust than expected.

== 8. Quantum codes and high-threshold codes <QUANTUM_CODES>

=== 8.1 Quantum codes vs classical codes
Quantum codes (especially CSS codes) are usually built from two classical codes: one corrects X errors, the other corrects Z errors.
The core constraint is the **no-cloning theorem**: we cannot copy qubits to check errors, we can only measure stabilizers (parity checks) to obtain information indirectly.

=== 8.2 What is a threshold?
This is a key metric for code performance.
- If the physical error rate $p < p_text("th")$, then as code length $N arrow infinity$, the logical error rate $P_L arrow 0$.
- If $p > p_text("th")$, increasing code length increases logical error rate.

=== 8.3 Common high-threshold codes

#table(
  columns: (1fr, 2fr, 2fr),
  inset: 10pt,
  align: horizon,
  stroke: 0.5pt + gray,
  table.header([*Type*], [*Representative: Surface Code*], [*Representative: quantum LDPC codes*]),
  [Structure], [2D grid, only nearest-neighbor interactions], [Sparse random graphs with complex long-range connections],
  [Threshold], [High ($approx 1\%$ under circuit noise)], [Medium/high (depends on construction)],
  [Code rate], [Very low ($1/N$, decreases with size)], [Finite constant ($k/N$ stays constant)],
  [Decoding difficulty], [Lower (MWPM, Union-Find)], [Higher (requires BP+OSD)]
)

== 9. The loop problem (Short Cycles / Loops) <LOOP_PROBLEM>

This is the biggest enemy of BP on quantum codes.

#note("Why are loops deadly?")[
  BP assumes the graph is **tree-like** (no cycles).
  
  1. **Echo chamber effect**:
     Suppose node A tells node B: "I think I have an error".
     If there is a loop $A arrow B arrow C arrow A$, that message comes back to A as: "C says you might also be in error".
     A mistakes this as independent confirmation and becomes overconfident.
  
  2. **Positive feedback oscillation**:
     This self-reinforcing loop causes probabilities to oscillate between 0 and 1, preventing convergence.
]

*The QEC dilemma*:
Degenerate quantum codes (especially topological codes like the surface code) **intrinsically contain many short loops** (because stabilizers must commute, which geometrically forms closed shapes). This makes standard BP perform poorly and forces the use of GBP (handling loops) or OSD (breaking loop effects).

== 10. BP algorithm explained (an intuitive view for beginners) <BP_ALGORITHM>

Belief Propagation (BP), also called the sum-product algorithm, is essentially a **"telephone game"**.

=== 10.1 Core roles
- **Variable nodes (V)**: physical qubits. They want to know if they are in error (0 or 1).
- **Check nodes (C)**: stabilizers. They enforce parity checks (e.g., an even number of 1s among neighbors).

=== 10.2 Algorithm flow (iterative)

We can understand it using **log-likelihood ratios (LLR)**: positive means a tendency toward 0, negative means a tendency toward 1.

1.  **Initialization (Input)**:
    Each variable node $V$ has an initial belief (prior) based on channel noise (e.g., error rate 0.1%).

2.  **Check node update ("work with me")**:
    Check node $C$ tells each connected variable $V_i$:
    *"Based on the states of the other variables connected to me, what state should you be in to satisfy my parity check?"*
    
    *Rule*: If the other variables are confident about 0, you must comply. If they are uncertain, your message is also uncertain.

3.  **Variable node update ("I listen to everyone")**:
    Variable node $V$ collects suggestions from all connected checks $C_k$, adds its own prior, and simply **sums** them.
    *"C1 says I might be 1, C2 says I am definitely 0, and I already think I am 0, so my combined belief is..."*

4.  **Decision**:
    Check the final LLR sign. If negative, declare the bit flipped.

5.  **Syndrome check**:
    Check whether the current correction satisfies all parity checks. If yes, stop; otherwise return to step 2 (until timeout).

#block(fill: blue.lighten(95%), stroke: blue, inset: 10pt, radius: 4pt)[
  *One-sentence summary of BP*:
  Everyone (qubits) keeps updating their belief based on neighbors (checks) until the network reaches consensus.
]


// Keep the previous note function definition
#let note(title, body) = {
  block(
    fill: luma(245),
    stroke: (left: 4pt + blue.darken(20%)),
    inset: 12pt,
    radius: (right: 4pt),
    width: 100%,
    [
      #text(weight: "bold", fill: blue.darken(20%), size: 1.1em)[#title]
      #v(0.5em)
      #body
    ]
  )
}

// ==========================================
// Drawing helper functions (for Surface Code)
// ==========================================
#let draw-surface-code() = {
  block(
    width: 100%, height: 160pt, stroke: 0.5pt + gray, radius: 5pt, inset: 10pt,
    {
      place(top + center, text(weight: "bold")[Surface Code (Rotated Surface Code) diagram])
      
      // Parameters
      let size = 40pt
      let offset-x = 80pt
      let offset-y = 40pt
      
      // Draw stabilizers (squares)
      // Z-check (green), X-check (orange)
      for r in range(2) {
        for c in range(2) {
          let x = offset-x + c * size
          let y = offset-y + r * size
          let color = if calc.even(r + c) { green.lighten(60%) } else { orange.lighten(60%) }
          let label = if calc.even(r + c) { "Z" } else { "X" }
          
          place(top + left, dx: x, dy: y, 
            rect(width: size, height: size, fill: color, stroke: 0.5pt + black)[
              #align(center + horizon, text(fill: black, size: 8pt, label))
            ]
          )
        }
      }
      
      // Draw data qubits (dots at vertices)
      for r in range(3) {
        for c in range(3) {
          let x = offset-x + c * size
          let y = offset-y + r * size
          place(top + left, dx: x - 4pt, dy: y - 4pt, 
            circle(radius: 4pt, fill: black)
          )
        }
      }
      
      // Legend
      place(top + left, dx: 10pt, dy: 100pt, block(width: 80pt)[
        #set text(size: 8pt)
        #stack(dir: ltr, spacing: 5pt, circle(radius: 3pt, fill: black), [Data qubits])
        #v(3pt)
        #stack(dir: ltr, spacing: 5pt, rect(width: 6pt, height: 6pt, fill: green.lighten(60%)), [Z check (face)])
        #v(3pt)
        #stack(dir: ltr, spacing: 5pt, rect(width: 6pt, height: 6pt, fill: orange.lighten(60%)), [X check (star)])
      ])

      // Draw an error example
      place(top + left, dx: offset-x + size - 6pt, dy: offset-y + size - 6pt, 
        text(fill: red, weight: "bold", size: 14pt)[$times$]
      )
      place(top + left, dx: 200pt, dy: 40pt, block(width: 120pt)[
        #set text(size: 9pt)
        *Error detection mechanism:* \
        The central data qubit has an error (red cross), which triggers both adjacent $X$ and $Z$ checks. \
        These paired "hot spots" are the **syndrome**.
      ])
    }
  )
}

// ==========================================
// Drawing helper functions (for LDPC matrix)
// ==========================================
#let draw-ldpc-matrix() = {
  block(
    width: 100%, height: 140pt, stroke: 0.5pt + gray, radius: 5pt, inset: 10pt,
    {
      place(top + center, text(weight: "bold")[LDPC sparse matrix $H$ diagram])
      
      let rows = 6
      let cols = 12
      let cell-size = 15pt
      let start-x = 30pt
      let start-y = 30pt
      
      // Draw grid box
      place(top + left, dx: start-x, dy: start-y, 
        rect(width: cols * cell-size, height: rows * cell-size, stroke: 1pt + black)
      )
      
      // Manually specify some points to show sparsity
      let points = (
        (0,0), (0,1), (0,4), (1,1), (1,5), (1,8), 
        (2,2), (2,3), (2,6), (3,0), (3,4), (3,9),
        (4,7), (4,8), (4,10), (5,5), (5,10), (5,11)
      )
      
      for p in points {
        let (r, c) = p
        place(top + left, dx: start-x + c * cell-size, dy: start-y + r * cell-size,
          rect(width: cell-size, height: cell-size, fill: blue.darken(30%))
        )
      }

      // Annotations
      place(top + left, dx: 250pt, dy: 40pt, block(width: 150pt)[
        #set text(size: 9pt)
        *Features:* \
        1. Most cells are empty (white). \
        2. Each row/column has only a few non-zero elements (blue). \
        3. This means each check involves only a few bits, and each bit participates in only a few checks.
      ])
    }
  )
}

// Main content

== 11. Surface Code <SURFACE_CODE>

The surface code is currently the most promising candidate for universal quantum computing. It is a typical **topological code**.

=== 11.1 Structure and diagram
Imagine a chessboard:
- **Data qubits**: located at the intersections (or edges) of the grid.
- **Stabilizers**: located in the plaquettes. Two types: one checks $Z$ errors (face operators), the other checks $X$ errors (vertex/star operators).


#draw-surface-code()

=== 11.2 Key properties
1.  **Locality**: the biggest engineering advantage. Each qubit only interacts with its neighbors. No need to connect the upper-left qubit to the lower-right.
2.  **High threshold**: about 1% physical error rate still correctable. Very forgiving.
3.  **Zero code rate**: the biggest downside.
    - No matter how large the board is, e.g., $1000 times 1000$ physical qubits, it typically encodes **one** logical qubit.
    - Resource overhead is huge.

=== 11.3 Why does BP perform poorly on the surface code?
From the diagram, each plaquette (check) shares data qubits with neighboring plaquettes. On the Tanner graph, this forms many **length-4 short cycles**.
- Standard BP cannot converge correctly in such dense short-cycle structures (positive feedback oscillation).
- Therefore surface codes typically use MWPM (minimum weight perfect matching) or Union-Find decoders.

== 12. LDPC codes (Low-Density Parity-Check Codes) <LDPC_CODES>

LDPC codes generalize surface codes. In classical communications (e.g., 5G, Wi-Fi) they are already standard. Quantum LDPC (qLDPC) is a hot research topic in recent years.

=== 12.1 What does "low density" mean?
"Low density" means the parity-check matrix $H$ is **sparse**.

#draw-ldpc-matrix()

- **Row weight**: number of bits involved in each check (usually a small constant like 6).
- **Column weight**: number of checks each bit participates in (also a small constant like 3 or 4).
- **Non-locality**: unlike the surface code, LDPC codes allow long-range connections. Points in the matrix can be far apart, not just neighbors.

=== 12.2 Why are qLDPC codes so exciting?
They solve the biggest pain of surface codes - **code rate**.

#note("Performance comparison")[
  - *Surface code*: uses $N$ physical qubits to encode $1$ logical qubit. $k/N arrow 0$.
  - *Good qLDPC codes*: use $N$ physical qubits to encode $k = N/10$ logical qubits. $k/N = "const"$.
  
  *Example*: to protect 100 logical qubits.
  - Surface code may need 100,000 physical qubits.
  - qLDPC may need 1,000 physical qubits.
]

=== 12.3 Decoding: BP's home field
Because qLDPC codes are typically constructed from expander graphs, they can still have cycles, but the cycles are usually long (large girth) or sufficiently random.
- **Standard BP** performs well on LDPC codes.
- **BP + OSD** is the standard decoding approach for qLDPC.

=== 12.4 Summary comparison table

#table(
  columns: (1fr, 2fr, 2fr),
  inset: 10pt,
  align: horizon,
  stroke: 0.5pt + gray,
  table.header([*Feature*], [*Surface Code*], [*Quantum LDPC code*]),
  [Geometry], [2D planar grid, local connections], [Complex network, long-range connections],
  [Short-cycle problem], [Very severe (many 4-cycles)], [Milder (designed to avoid short cycles)],
  [Encoding efficiency], [Very low ($k=1$)], [High (finite code rate)],
  [Applicable decoders], [MWPM, Union-Find (BP needs modifications)], [BP, BP+OSD],
  [Engineering difficulty], [Low (simple wiring)], [High (requires long wires/multi-layer routing)]
)

// Reuse the previous style definition
#let note(title, body) = {
  block(
    fill: luma(245),
    stroke: (left: 4pt + blue.darken(20%)),
    inset: 12pt,
    radius: (right: 4pt),
    width: 100%,
    [
      #text(weight: "bold", fill: blue.darken(20%), size: 1.1em)[#title]
      #v(0.5em)
      #body
    ]
  )
}

// ==========================================
// Drawing function: BP message passing diagram
// ==========================================
#let draw-bp-mechanism() = {
  block(
    width: 100%, height: 160pt, stroke: 0.5pt + gray, radius: 5pt, inset: 10pt,
    {
      place(top + center, text(weight: "bold")[Message passing on a Tanner graph])
      
      let v-y = 110pt
      let c-y = 40pt
      let v1-x = 40pt
      let v2-x = 120pt
      let c-x = 80pt

      // Draw lines
      place(top + left, line(start: (v1-x + 10pt, v-y), end: (c-x + 10pt, c-y + 20pt), stroke: 1pt + gray))
      place(top + left, line(start: (v2-x + 10pt, v-y), end: (c-x + 10pt, c-y + 20pt), stroke: 1pt + gray))

      // Draw variable nodes (circles)
      place(top + left, dx: v1-x, dy: v-y, circle(radius: 12pt, stroke: 1pt + black, fill: white)[$V_1$])
      place(top + left, dx: v2-x, dy: v-y, circle(radius: 12pt, stroke: 1pt + black, fill: white)[$V_2$])

      // Draw check node (square)
      place(top + left, dx: c-x, dy: c-y, rect(width: 24pt, height: 24pt, stroke: 1pt + black, fill: gray.lighten(80%))[$C_a$])

      // Draw message arrows
      // V -> C
      place(top + left, dx: 45pt, dy: 80pt, text(fill: blue, size: 9pt)[$m_(V arrow C)$])
      place(top + left, dx: 55pt, dy: 75pt, rotate(-60deg, text(fill: blue)[$arrow$]))
      
      // C -> V
      place(top + left, dx: 120pt, dy: 80pt, text(fill: red, size: 9pt)[$m_(C arrow V)$])
      place(top + left, dx: 115pt, dy: 75pt, rotate(60deg, text(fill: red)[$arrow.b$]))

      // Explanation
      place(top + left, dx: 180pt, dy: 40pt, block(width: 160pt)[
        #set text(size: 9pt)
        *Core principle (Extrinsic Principle)*:\
        The message $V_2$ sends to $C_a$ cannot include the information that $C_a$ just sent to $V_2$.\
        \
        *Plainly:*\
        "Apart from what you told me, here is what I think..."
      ])
    }
  )
}

// Main content begins

== 13. BP algorithm deep dive <BP_DETAILED_CODE>

Belief Propagation (BP) is the standard algorithm for exact inference on trees or approximate inference on graphs with cycles. In QEC, we often use **log-domain BP** for numerical stability.

=== 13.1 Why use log-likelihood ratios (LLR)?
Directly passing probabilities $p in [0, 1]$ can underflow to 0 after repeated multiplications. We define LLR:
$ L(x) = ln(P(x=0) / P(x=1)) $
- $L > 0$: more likely 0 (no error).
- $L < 0$: more likely 1 (error).
- $|L|$: confidence magnitude.
#pagebreak()
#draw-bp-mechanism()

=== 13.2 Core update formulas

The algorithm alternates message passing between variable nodes $V_i$ and check nodes $C_j$.

1.  **Variable to Check ($V arrow C$)**
    $V_i$ tells $C_j$ the probability of being 0 or 1.
    This equals its **initial observation** plus **all other check nodes** (except $C_j$) suggestions.

    $ m_(i arrow j) = L_i^"init" + sum_(k in N(i) backslash j) m_(k arrow i) $

2.  **Check to Variable ($C arrow V$)**
    $C_j$ tells $V_i$ what it must be to satisfy parity.
    This involves a nonlinear "tanh" operation (a soft version of XOR).

    $ m_(j arrow i) = 2 "tanh"^(-1) ( (-1)^(S_j) product_(k in N(j) backslash i) "tanh"(m_(k arrow j) / 2) ) $

    - $S_j in {0, 1}$ is the syndrome value of the check node.
    - If $S_j=1$ (check fails), the sign term $(-1)^1 = -1$ flips the message sign, encouraging a flip.
    - If some $m_{k \to j}$ is near 0 (uncertain), the product is near 0, and $C_j$ gives a near-zero suggestion.

=== 13.3 Code logic (Pythonic pseudocode)

In practice (C++ or Python), the code structure is typically:

```python
# Initialization
# channel_probs: physical error rate per bit
llr = log((1 - channel_probs) / channel_probs) # initial LLR
check_to_var_msg = zeros(num_checks, num_vars) # store C->V messages

for iter in range(max_iters):
    
    # --- Step 1: Variable node processing (V -> C) ---
    # For each edge (i, j), compute the message from V_i to C_j
    # Trick: compute total once, then subtract the message from j (O(deg) -> O(1))
    var_to_check_msg = zeros(num_checks, num_vars)
    for i in range(num_vars):
        incoming_sum = llr[i] + sum(check_to_var_msg[:, i]) # includes all C info
        for j in neighbors(i):
            # Extrinsic principle: total minus the message from j
            var_to_check_msg[j, i] = incoming_sum - check_to_var_msg[j, i]

    # --- Step 2: Check node processing (C -> V) ---
    # Core challenge: implement the tanh product formula
    # Optimization: usually use lookup tables or min-sum approximation
    for j in range(num_checks):
        syndrome_sign = -1 if syndrome[j] == 1 else 1
        
        # Compute tanh product over all variables connected to this check
        prod_tanh = 1.0
        for i in neighbors(j):
            prod_tanh *= tanh(var_to_check_msg[j, i] / 2)
            
        # Update C -> V messages
        for i in neighbors(j):
            # Extrinsic principle: divide by current variable's tanh (multiplicative inverse)
            # Note division by 0 (numerical stability)
            extrinsic_val = prod_tanh / tanh(var_to_check_msg[j, i] / 2)
            extrinsic_val = clip(extrinsic_val, -0.999999, 0.999999) 
            check_to_var_msg[j, i] = syndrome_sign * 2 * atanh(extrinsic_val)

    # --- Step 3: Soft decision and check ---
    new_llr = llr.copy()
    for i in range(num_vars):
        new_llr[i] += sum(check_to_var_msg[:, i])
    
    hard_decision = (new_llr < 0).astype(int) # LLR < 0 -> 1 (error)
    
    if (H @ hard_decision % 2 == syndrome).all():
        return hard_decision # converged
        
return fail # timeout```
