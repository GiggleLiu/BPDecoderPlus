// BP + OSD Algorithm for Quantum Error Correction
// Lecture Note based on arXiv:2005.07016
// "Decoding Across the Quantum LDPC Code Landscape"
// by Roffe, White, Burton, and Campbell (2020)

#import "@preview/cetz:0.3.2": canvas, draw

#set document(title: "BP + OSD Algorithm for Quantum Error Correction")
#set page(paper: "a4", margin: 2.5cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// Custom environments
#let definition(body) = block(
  width: 100%,
  stroke: (left: 3pt + blue),
  inset: (left: 12pt, y: 8pt),
  fill: rgb("#f0f7ff"),
  [#text(weight: "bold")[Definition.] #body]
)

#let example(body) = block(
  width: 100%,
  stroke: (left: 3pt + green),
  inset: (left: 12pt, y: 8pt),
  fill: rgb("#f0fff0"),
  [#text(weight: "bold")[Example.] #body]
)

#let notation(body) = block(
  width: 100%,
  stroke: (left: 3pt + purple),
  inset: (left: 12pt, y: 8pt),
  fill: rgb("#f8f0ff"),
  [#text(weight: "bold")[Notation.] #body]
)

#let keypoint(body) = block(
  width: 100%,
  stroke: 1pt + orange,
  inset: 10pt,
  radius: 4pt,
  fill: rgb("#fffaf0"),
  [#text(weight: "bold")[Key Point.] #body]
)

// Title
#align(center)[
  #text(size: 18pt, weight: "bold")[
    BP + OSD Algorithm for Quantum Error Correction
  ]
  #v(0.5em)
  #text(size: 12pt)[Lecture Note based on arXiv:2005.07016]
  #v(0.5em)
  #text(size: 10pt, style: "italic")[
    Roffe, White, Burton, and Campbell (2020)
  ]
]

#v(1em)

#outline(indent: auto, depth: 2)

#pagebreak()

= Introduction

== Overview

This lecture note introduces the *BP+OSD decoder* for quantum error correction:

- *BP* = Belief Propagation (a classical decoding algorithm)
- *OSD* = Ordered Statistics Decoding (a post-processing technique)

Together, BP+OSD provides a general-purpose decoder for *quantum LDPC codes* (Low-Density Parity Check codes).

== Learning Objectives

By the end of this note, you will understand:

#enum(
  [How classical error correction codes work],
  [The Belief Propagation algorithm for decoding],
  [Why BP fails for quantum codes (the degeneracy problem)],
  [How OSD fixes the degeneracy problem],
  [The complete BP+OSD decoding algorithm],
)

#pagebreak()

= Preliminaries

== Binary Arithmetic

#notation[
  All arithmetic in this note is performed in *binary* (modulo 2):
  - $0 + 0 = 0$, $quad$ $1 + 0 = 0 + 1 = 1$, $quad$ $1 + 1 = 0$
  - This is also written as XOR: $a plus.o b = (a + b) mod 2$
  - Vectors and matrices use element-wise mod-2 arithmetic
]

== Hamming Weight and Distance

#definition[
  The *Hamming weight* of a binary vector $bold(v)$ is the number of 1s it contains:
  $ |bold(v)| = sum_i v_i $

  The *Hamming distance* between two vectors $bold(u)$ and $bold(v)$ is the number of positions where they differ:
  $ d(bold(u), bold(v)) = |bold(u) + bold(v)| $
]

#example[
  For $bold(v) = (1, 0, 1, 1, 0)$: Hamming weight $|bold(v)| = 3$

  For $bold(u) = (1, 1, 0, 1, 0)$ and $bold(v) = (1, 0, 1, 1, 0)$:
  - $bold(u) + bold(v) = (0, 1, 1, 0, 0)$
  - Hamming distance $d(bold(u), bold(v)) = 2$
]

#pagebreak()

= Classical Error Correction

== Linear Codes

#definition[
  An *$[n, k, d]$ linear code* $cal(C)$ is a set of binary vectors (called *codewords*) where:
  - $n$ = *block length* (number of bits in each codeword)
  - $k$ = *dimension* (number of information bits encoded)
  - $d$ = *minimum distance* (minimum Hamming weight among non-zero codewords)
  - *Rate* $R = k\/n$ (fraction of information bits)
]

== Parity Check Matrix

A linear code can be defined by an $m times n$ *parity check matrix* $H$.

#definition[
  A binary vector $bold(c)$ is a *codeword* if and only if:
  $ H dot bold(c) = bold(0) $
  where $bold(0)$ is the all-zeros vector and arithmetic is mod-2.

  We write the code as $cal(C)_H$ to indicate it is defined by matrix $H$.
]

#notation[
  - $H_(i j)$ denotes the entry in row $i$, column $j$ of matrix $H$
  - $m$ = number of rows in $H$ (number of parity checks)
  - $n$ = number of columns in $H$ (number of bits)
  - $"rank"(H)$ = number of linearly independent rows
  - By the rank-nullity theorem: $k = n - "rank"(H)$
]

#example[
  The *$[3, 1, 3]$ repetition code* encodes 1 bit into 3 bits by triplication.

  Parity check matrix:
  $ H = mat(1, 1, 0; 0, 1, 1) $

  Verification: $H dot mat(0;0;0) = mat(0;0)$ ✓ and $H dot mat(1;1;1) = mat(0;0)$ ✓

  So the codewords are: $cal(C)_H = {(0,0,0), (1,1,1)}$

  Parameters: $n = 3$ bits, $k = 3 - 2 = 1$ info bit, $d = 3$ (weight of $(1,1,1)$)
]

#pagebreak()

== Error Model and Syndrome

#definition[
  In the *binary symmetric channel* (BSC) with error probability $p$:
  - Each bit is independently flipped with probability $p$
  - Original codeword: $bold(c)$
  - Error pattern: $bold(e)$ (a binary vector, $e_i = 1$ means bit $i$ flipped)
  - Received word: $bold(r) = bold(c) + bold(e)$
]

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Channel diagram
    rect((-4, -0.5), (-2, 0.5), name: "input")
    content("input", $bold(c)$)

    rect((0, -0.7), (2, 0.7), name: "channel", fill: rgb("#f0f0f0"))
    content("channel", [BSC($p$)])

    rect((4, -0.5), (6, 0.5), name: "output")
    content("output", $bold(r) = bold(c) + bold(e)$)

    // Arrows
    line((-2, 0), (-0.1, 0), mark: (end: ">"))
    line((2.1, 0), (4, 0), mark: (end: ">"))

    // Error annotation
    content((1, -1.3), text(size: 9pt)[$bold(e)$: random error])
  }),
  caption: [Binary symmetric channel model]
)

#definition[
  The *syndrome* of a received word $bold(r)$ is:
  $ bold(s) = H dot bold(r) $

  Since $H dot bold(c) = bold(0)$ for any codeword, we have:
  $ bold(s) = H dot bold(r) = H dot (bold(c) + bold(e)) = H dot bold(c) + H dot bold(e) = bold(0) + H dot bold(e) = H dot bold(e) $
]

#keypoint[
  The syndrome depends *only on the error*, not on which codeword was sent!
  This is what makes syndrome-based decoding possible.
]

== The Decoding Problem

Given: Parity check matrix $H$ and syndrome $bold(s) = H dot bold(e)$

Find: The most likely error $bold(e)^*$ that could have produced $bold(s)$

#definition[
  *Maximum likelihood decoding* finds:
  $ bold(e)^* = arg min_(bold(e) : H dot bold(e) = bold(s)) |bold(e)| $

  That is, the minimum Hamming weight error consistent with the syndrome.
]

#pagebreak()

== Factor Graphs

To understand the Belief Propagation algorithm, we need the concept of *factor graphs*.

#definition[
  A *factor graph* is a bipartite graph $G = (V, U, E)$ representing the parity check matrix $H$:

  - *Data nodes* (also called *variable nodes*):
    $ V = {v_1, v_2, ..., v_n} $
    One node $v_j$ for each column of $H$ (each bit position)

  - *Parity nodes* (also called *check nodes* or *factor nodes*):
    $ U = {u_1, u_2, ..., u_m} $
    One node $u_i$ for each row of $H$ (each parity check)

  - *Edges*: $E = {(v_j, u_i) : H_(i j) = 1}$
    An edge connects $v_j$ to $u_i$ if bit $j$ participates in check $i$
]

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Legend
    circle((-3, 0), radius: 0.3, name: "leg-data")
    content(((-1.5, 0)), [= Data node (bit)])

    rect((3, -0.3), (3.6, 0.3), name: "leg-parity")
    content(((5.2, 0)), [= Parity node (check)])
  }),
  caption: [Factor graph node conventions: circles for data nodes, squares for parity nodes]
)

#v(0.5em)

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Data nodes (top row)
    circle((0, 2), radius: 0.35, name: "v1")
    content("v1", $v_1$)
    circle((2, 2), radius: 0.35, name: "v2")
    content("v2", $v_2$)
    circle((4, 2), radius: 0.35, name: "v3")
    content("v3", $v_3$)

    // Parity nodes (bottom row)
    rect((0.7, -0.3), (1.3, 0.3), name: "u1")
    content("u1", $u_1$)
    rect((2.7, -0.3), (3.3, 0.3), name: "u2")
    content("u2", $u_2$)

    // Edges for H = [[1,1,0], [0,1,1]]
    line((0, 1.65), (1, 0.3))      // v1 - u1
    line((2, 1.65), (1, 0.3))      // v2 - u1
    line((2, 1.65), (3, 0.3))      // v2 - u2
    line((4, 1.65), (3, 0.3))      // v3 - u2

    // Labels
    content((2, -1.2), text(size: 9pt)[Factor graph for $H = mat(1,1,0; 0,1,1)$])
  }),
  caption: [Example factor graph for the $[3,1,3]$ repetition code]
)

#notation[
  - $V(u_i) = {v_j : H_(i j) = 1}$ = set of data nodes connected to parity node $u_i$
  - $U(v_j) = {u_i : H_(i j) = 1}$ = set of parity nodes connected to data node $v_j$
  - These are called the *neighborhoods* of the nodes
]

#pagebreak()

== Low-Density Parity Check (LDPC) Codes

#definition[
  An *$(l, q)$-LDPC code* is a linear code whose parity check matrix $H$ satisfies:
  - Each column has at most $l$ ones (each bit is in at most $l$ checks)
  - Each row has at most $q$ ones (each check involves at most $q$ bits)

  The matrix $H$ is called *sparse* because $l$ and $q$ are small constants independent of $n$.
]

#keypoint[
  LDPC codes are important because their sparse structure enables efficient decoding via Belief Propagation with complexity $O(n)$ per iteration.
]

#pagebreak()

= Belief Propagation (BP) Algorithm

== Overview

*Belief Propagation* (BP), also called the *sum-product algorithm*, is an iterative message-passing algorithm on the factor graph.

#definition[
  The goal of BP is to compute, for each bit $j$, the *marginal probability*:
  $ P_1(e_j) = P(e_j = 1 | bold(s)) $

  This is called a *soft decision* -- it tells us how likely each bit is to be flipped.
]

#notation[
  - $p$ = channel error probability (probability each bit flips)
  - $m_(v_j arrow.r u_i)$ = message from data node $v_j$ to parity node $u_i$
  - $m_(u_i arrow.r v_j)$ = message from parity node $u_i$ to data node $v_j$
  - Messages represent beliefs about whether $e_j = 1$
]

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Data node
    circle((0, 0), radius: 0.4, name: "vj")
    content("vj", $v_j$)

    // Parity node
    rect((4, -0.35), (4.7, 0.35), name: "ui")
    content("ui", $u_i$)

    // Messages
    line((0.5, 0.15), (3.9, 0.15), mark: (end: ">"), stroke: blue)
    content((2.2, 0.55), text(fill: blue, size: 9pt)[$m_(v_j arrow.r u_i)$])

    line((3.9, -0.15), (0.5, -0.15), mark: (end: ">"), stroke: red)
    content((2.2, -0.55), text(fill: red, size: 9pt)[$m_(u_i arrow.r v_j)$])
  }),
  caption: [Messages passed between data and parity nodes]
)

== Log-Likelihood Ratios (LLR)

#definition[
  Instead of probabilities, BP uses *log-likelihood ratios* (LLR) for numerical stability:
  $ "LLR"(e_j) = log (P(e_j = 0)) / (P(e_j = 1)) $

  - $"LLR" > 0$ means $e_j = 0$ is more likely (bit probably correct)
  - $"LLR" < 0$ means $e_j = 1$ is more likely (bit probably flipped)
  - $|"LLR"|$ indicates confidence level
]

For the channel with error probability $p$, the *channel LLR* is:
$ p_l = log (1 - p) / p $

Since $p < 0.5$ in practice, we have $p_l > 0$.

#pagebreak()

== BP Algorithm: Step-by-Step

#let step-box(num, title, content) = {
  box(
    width: 100%,
    stroke: 0.5pt + gray,
    inset: 10pt,
    radius: 4pt,
    [
      #text(weight: "bold")[Step #num: #title]
      #v(0.3em)
      #content
    ]
  )
}

#step-box(1, "Initialization")[
  Set the channel LLR:
  $ p_l = log (1-p) / p $

  Initialize all messages from data nodes to parity nodes with the channel prior:
  $ m_(v_j arrow.r u_i) := p_l quad "for all edges" (v_j, u_i) $
]

#v(0.5em)

#step-box(2, "Parity-to-Data Messages")[
  Each parity node $u_i$ sends a message to each connected data node $v_j$:

  $ m_(u_i arrow.r v_j) = (-1)^(s_i) dot alpha dot product_(v'_j in V(u_i) backslash v_j) "sign"(m_(v'_j arrow.r u_i)) dot min_(v'_j in V(u_i) backslash v_j) |m_(v'_j arrow.r u_i)| $

  Where:
  - $s_i$ = the $i$-th syndrome bit (given as input, either 0 or 1)
  - $V(u_i) backslash v_j$ = all neighbors of $u_i$ except $v_j$ (defined in Section 3.5)
  - $"sign"(x) = +1$ if $x >= 0$, else $-1$
  - $alpha = 1 - 2^(-t)$ is a *damping factor* at iteration $t$ (helps convergence)
]

#v(0.5em)

#step-box(3, "Data-to-Parity Messages")[
  Each data node $v_j$ sends a message to each connected parity node $u_i$:

  $ m_(v_j arrow.r u_i) = p_l + sum_(u'_i in U(v_j) backslash u_i) m_(u'_i arrow.r v_j) $

  Where $U(v_j) backslash u_i$ = all parity neighbors of $v_j$ except $u_i$.
]

#pagebreak()

#step-box(4, "Compute Soft Decisions")[
  For each bit $j$, compute the total belief (sum of all evidence):

  $ P_1(e_j) = p_l + sum_(u_i in U(v_j)) m_(u_i arrow.r v_j) $
]

#v(0.5em)

#step-box(5, "Make Hard Decisions")[
  Convert soft decisions to a binary estimate:

  $ e_j^"BP" = cases(
    1 & "if" P_1(e_j) < 0 quad "(more likely flipped)",
    0 & "otherwise" quad "(more likely correct)"
  ) $

  This gives us the BP estimate $bold(e)^"BP" = (e_1^"BP", e_2^"BP", ..., e_n^"BP")$.
]

#v(0.5em)

#step-box(6, "Check Convergence")[
  Verify if the estimate satisfies the syndrome equation:

  $ H dot bold(e)^"BP" = bold(s) quad ? $

  - *If yes:* BP has *converged*. Return $bold(e)^"BP"$ and soft decisions $P_1$.
  - *If no:* Go back to Step 2 and repeat.
  - *If max iterations reached:* BP has *failed to converge*.
]

#pagebreak()

== BP Algorithm: Pseudocode

#figure(
  align(left)[
    #box(
      width: 100%,
      stroke: 1pt,
      inset: 12pt,
      radius: 4pt,
      fill: luma(250),
      [
        #text(weight: "bold", size: 10pt)[Algorithm 1: Belief Propagation (Min-Sum Variant)]
        #v(0.5em)
        #text(size: 9pt)[
          ```
          Input: Parity check matrix H, syndrome s, error probability p
          Output: (converged, error_estimate, soft_decisions)

          function BP(H, s, p, max_iter=n):
              p_l = log((1-p)/p)                    // Channel LLR

              // Step 1: Initialize all messages
              for each edge (v_j, u_i) where H[i,j] = 1:
                  m[v_j → u_i] = p_l

              for t = 1 to max_iter:
                  α = 1 - 2^(-t)                    // Damping factor

                  // Step 2: Parity-to-Data messages
                  for each parity node u_i:
                      for each neighbor v_j of u_i:
                          others = V(u_i) \ {v_j}   // All neighbors except v_j
                          sign_prod = (-1)^(s[i]) × ∏_{v' in others} sign(m[v'→u_i])
                          min_mag = min_{v' in others} |m[v'→u_i]|
                          m[u_i → v_j] = α × sign_prod × min_mag

                  // Step 3: Data-to-Parity messages
                  for each data node v_j:
                      for each neighbor u_i of v_j:
                          others = U(v_j) \ {u_i}   // All neighbors except u_i
                          m[v_j → u_i] = p_l + Σ_{u' in others} m[u'→v_j]

                  // Steps 4-5: Compute decisions
                  for j = 1 to n:
                      P_1[j] = p_l + Σ_{u_i in U(v_j)} m[u_i→v_j]
                      e_BP[j] = 1 if P_1[j] < 0 else 0

                  // Step 6: Check convergence
                  if H × e_BP == s:
                      return (True, e_BP, P_1)

              return (False, e_BP, P_1)
          ```
        ]
      ]
    )
  ],
  caption: [Belief Propagation pseudocode]
)

#pagebreak()

= Quantum Error Correction Basics

== Qubits and Quantum States

#definition[
  A *qubit* is a quantum two-level system. Its state is written using *ket notation*:
  $ |psi〉 = alpha |0〉 + beta |1〉 $

  where:
  - $|0〉 = mat(1; 0)$ and $|1〉 = mat(0; 1)$ are the *computational basis states*
  - $alpha, beta$ are complex numbers with $|alpha|^2 + |beta|^2 = 1$
  - The ket symbol $|dot〉$ is standard notation for quantum states
]

#notation[
  Common quantum states:
  - $|0〉, |1〉$ = computational basis
  - $|+〉 = 1/sqrt(2)(|0〉 + |1〉)$ = superposition (plus state)
  - $|-〉 = 1/sqrt(2)(|0〉 - |1〉)$ = superposition (minus state)
]

== Pauli Operators

#definition[
  The *Pauli operators* are the fundamental single-qubit error operations:

  #figure(
    table(
      columns: 4,
      align: center,
      [*Symbol*], [*Matrix*], [*Binary repr.*], [*Effect on states*],
      [$bb(1)$ (Identity)], [$mat(1,0;0,1)$], [$(0,0)$], [No change],
      [$X$ (bit flip)], [$mat(0,1;1,0)$], [$(1,0)$], [$|0〉 arrow.l.r |1〉$],
      [$Z$ (phase flip)], [$mat(1,0;0,-1)$], [$(0,1)$], [$|+〉 arrow.l.r |-〉$],
      [$Y = i X Z$], [$mat(0,-i;i,0)$], [$(1,1)$], [Both flips],
    ),
    caption: [Pauli operators]
  )
]

#keypoint[
  Quantum errors are modeled as random Pauli operators:
  - *X errors* = bit flips (like classical errors)
  - *Z errors* = phase flips (uniquely quantum, no classical analogue)
  - *Y errors* = both (can be written as $Y = i X Z$)
]

#pagebreak()

== Binary Representation of Pauli Errors

#definition[
  An $n$-qubit Pauli error $E$ can be written in *binary representation*:
  $ E arrow.r.bar bold(e)_Q = (bold(x), bold(z)) $

  where:
  - $bold(x) = (x_1, ..., x_n)$ indicates X components ($x_j = 1$ means X error on qubit $j$)
  - $bold(z) = (z_1, ..., z_n)$ indicates Z components ($z_j = 1$ means Z error on qubit $j$)
]

#example[
  The error $E = X_1 Z_3$ on 3 qubits (X on qubit 1, Z on qubit 3):
  $ bold(e)_Q = (bold(x), bold(z)) = ((1,0,0), (0,0,1)) $
]

== CSS Codes

#definition[
  A *CSS code* (Calderbank-Shor-Steane code) is a quantum error-correcting code with a structure that allows X and Z errors to be corrected independently.

  A CSS code is defined by two classical parity check matrices $H_X$ and $H_Z$ satisfying:
  $ H_X dot H_Z^T = bold(0) quad ("orthogonality constraint") $

  The combined quantum parity check matrix is:
  $ H_"CSS" = mat(H_Z, bold(0); bold(0), H_X) $
]

#keypoint[
  The orthogonality constraint $H_X dot H_Z^T = bold(0)$ ensures that the quantum stabilizers *commute* (a necessary condition for valid quantum codes).
]

#pagebreak()

== Syndrome Measurement in CSS Codes

For a CSS code with error $E arrow.r.bar bold(e)_Q = (bold(x), bold(z))$:

#definition[
  The *quantum syndrome* is:
  $ bold(s)_Q = (bold(s)_x, bold(s)_z) = (H_Z dot bold(x), H_X dot bold(z)) $

  - $bold(s)_x = H_Z dot bold(x)$ detects X (bit-flip) errors
  - $bold(s)_z = H_X dot bold(z)$ detects Z (phase-flip) errors
]

#figure(
  canvas(length: 1cm, {
    import draw: *

    // X-error decoding
    rect((-4, 0.8), (-0.5, 2), fill: rgb("#e8f4e8"), name: "xbox")
    content((-2.25, 1.6), [X-error decoding])
    content((-2.25, 1.1), text(size: 9pt)[$H_Z dot bold(x) = bold(s)_x$])

    // Z-error decoding
    rect((0.5, 0.8), (4, 2), fill: rgb("#e8e8f4"), name: "zbox")
    content((2.25, 1.6), [Z-error decoding])
    content((2.25, 1.1), text(size: 9pt)[$H_X dot bold(z) = bold(s)_z$])

    // Label
    content((0, 0.2), text(size: 9pt)[Two independent classical problems!])
  }),
  caption: [CSS codes allow independent X and Z decoding]
)

#keypoint[
  CSS codes allow *independent decoding*:
  - Decode X errors using matrix $H_Z$ and syndrome $bold(s)_x$
  - Decode Z errors using matrix $H_X$ and syndrome $bold(s)_z$

  Each is a classical syndrome decoding problem — so BP can be applied!
]

== Quantum Code Parameters

#notation[
  Quantum codes use double-bracket notation $[[n, k, d]]$:
  - $n$ = number of physical qubits
  - $k$ = number of logical qubits encoded
  - $d$ = code distance (minimum weight of undetectable errors)

  Compare to classical $[n, k, d]$ notation (single brackets).
]

#definition[
  A *quantum LDPC (QLDPC) code* is a CSS code where $H_"CSS"$ is sparse.

  An *$(l_Q, q_Q)$-QLDPC code* has:
  - Each column of $H_"CSS"$ has at most $l_Q$ ones
  - Each row of $H_"CSS"$ has at most $q_Q$ ones
]

#pagebreak()

== The Hypergraph Product Construction

#definition[
  The *hypergraph product* constructs a quantum CSS code from a classical code.

  Given classical code with $m times n$ parity check matrix $H$:

  $ H_X = mat(H times.o bb(1)_n, bb(1)_m times.o H^T) $
  $ H_Z = mat(bb(1)_n times.o H, H^T times.o bb(1)_m) $

  Where:
  - $times.o$ = *Kronecker product* (tensor product of matrices)
  - $bb(1)_n$ = $n times n$ identity matrix
  - $H^T$ = transpose of $H$
]

#example[
  *Toric Code:* Hypergraph product of the ring code (cyclic repetition code).

  From classical $[n, 1, n]$ ring code $arrow.r$ quantum $[[2n^2, 2, n]]$ Toric code.

  Properties:
  - $(4, 4)$-QLDPC: each stabilizer involves at most 4 qubits
  - High threshold (~10.3% with optimal decoder)
  - Rate $R = 2/(2n^2) arrow.r 0$ as $n arrow.r infinity$
]

#pagebreak()

= The Degeneracy Problem

== Why BP Fails on Quantum Codes

#box(
  width: 100%,
  stroke: 2pt + red,
  inset: 12pt,
  radius: 4pt,
  fill: rgb("#fff5f5"),
  [
    #text(weight: "bold", fill: red)[The Degeneracy Problem]
    #v(0.5em)

    In quantum codes, *multiple different errors can produce the same syndrome*.

    This is called *degeneracy* and it breaks BP!
  ]
)

#definition[
  Two errors $bold(e)_1$ and $bold(e)_2$ are *degenerate* if:
  $ H dot bold(e)_1 = H dot bold(e)_2 = bold(s) $

  In quantum codes, degenerate errors are often *equivalent* for error correction purposes.
]

== The Split-Belief Problem

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Two solutions
    circle((-2, 0), radius: 0.6, fill: rgb("#ffe0e0"), name: "e1")
    content("e1", $bold(e)_1$)

    circle((2, 0), radius: 0.6, fill: rgb("#e0e0ff"), name: "e2")
    content("e2", $bold(e)_2$)

    // Same syndrome
    rect((-0.5, -2.5), (0.5, -1.7), name: "syn")
    content("syn", $bold(s)$)

    // Arrows
    line((-1.5, -0.5), (-0.3, -1.6), mark: (end: ">"))
    line((1.5, -0.5), (0.3, -1.6), mark: (end: ">"))

    // Labels
    content((0, 0.3), text(size: 9pt)[Equal weight])
    content((0, -3.2), text(size: 9pt)[Same syndrome!])
  }),
  caption: [Two errors with the same syndrome cause BP to fail]
)

When BP encounters degenerate errors of equal weight:

#enum(
  [BP assigns high probability to *both* solutions $bold(e)_1$ and $bold(e)_2$],
  [The beliefs "split" between the two solutions],
  [BP outputs $bold(e)^"BP" approx bold(e)_1 + bold(e)_2$],
  [Check: $H dot bold(e)^"BP" = H dot (bold(e)_1 + bold(e)_2) = bold(s) + bold(s) = bold(0) eq.not bold(s)$],
  [*BP fails to converge!*]
)

#keypoint[
  For the Toric code, degeneracy is so prevalent that *BP alone shows no threshold* — increasing code distance makes performance worse, not better!
]

#pagebreak()

= Ordered Statistics Decoding (OSD)

== The Key Insight

The parity check matrix $H$ (size $m times n$ with $n > m$) has more columns than rows and cannot be directly inverted.

#keypoint[
  We can select a subset of $r = "rank"(H)$ linearly independent columns to form an invertible $m times r$ submatrix!
]

#definition[
  For an $m times n$ matrix $H$ with $"rank"(H) = r$:
  - *Basis set* $[S]$: indices of $r$ linearly independent columns
  - *Remainder set* $[T]$: indices of the remaining $k' = n - r$ columns
  - $H_([S])$: the $m times r$ submatrix of columns in $[S]$ (this is invertible!)
  - $H_([T])$: the $m times k'$ submatrix of columns in $[T]$
]

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Original matrix
    rect((-4, -1), (-1, 1), fill: rgb("#f0f0f0"), name: "H")
    content("H", $H$)
    content((-2.5, -1.5), text(size: 9pt)[$m times n$])

    // Arrow
    line((-0.5, 0), (0.5, 0), mark: (end: ">"))
    content((0, 0.5), text(size: 8pt)[split])

    // Basis submatrix
    rect((1, -1), (2.5, 1), fill: rgb("#e0ffe0"), name: "HS")
    content("HS", $H_([S])$)
    content((1.75, -1.5), text(size: 9pt)[$m times r$])
    content((1.75, -2), text(size: 8pt)[invertible!])

    // Remainder submatrix
    rect((3, -1), (5, 1), fill: rgb("#ffe0e0"), name: "HT")
    content("HT", $H_([T])$)
    content((4, -1.5), text(size: 9pt)[$m times k'$])
  }),
  caption: [Splitting $H$ into basis and remainder parts]
)

#pagebreak()

== OSD-0: The Basic Algorithm

#definition[
  *OSD-0* (zeroth-order OSD) finds a solution by:
  1. Choosing a "good" basis $[S]$ using BP soft decisions $P_1$
  2. Solving for the basis bits via matrix inversion
  3. Setting all remainder bits to zero
]

#figure(
  align(left)[
    #box(
      width: 100%,
      stroke: 1pt,
      inset: 12pt,
      radius: 4pt,
      fill: luma(250),
      [
        #text(weight: "bold")[Algorithm 2: OSD-0]
        #v(0.5em)

        *Input:*
        - Parity matrix $H$ (size $m times n$, rank $r$)
        - Syndrome $bold(s)$
        - BP soft decisions $P_1(e_1), ..., P_1(e_n)$ (from Algorithm 1)

        *Steps:*

        #enum(
          [*Rank bits by probability:*
           Sort bit indices by $P_1$ values: most-likely-flipped first.
           Result: ordered list $[O_"BP"] = (j_1, j_2, ..., j_n)$],

          [*Reorder columns:*
           $H_([O_"BP"])$ = matrix $H$ with columns reordered by $[O_"BP"]$],

          [*Select basis:*
           Scan left-to-right, select first $r$ linearly independent columns.
           Basis indices: $[S]$. Remainder indices: $[T]$ (size $k' = n - r$)],

          [*Solve on basis:*
           $bold(e)_([S]) = H_([S])^(-1) dot bold(s)$],

          [*Set remainder to zero:*
           $bold(e)_([T]) = bold(0)$ (zero vector of length $k'$)],

          [*Remap to original ordering:*
           Combine $(bold(e)_([S]), bold(e)_([T]))$ and undo the permutation]
        )

        *Output:* $bold(e)^"OSD-0"$ satisfying $H dot bold(e)^"OSD-0" = bold(s)$
      ]
    )
  ],
  caption: [OSD-0 algorithm]
)

#pagebreak()

== Why OSD Resolves Degeneracy

#keypoint[
  OSD resolves split beliefs by *forcing a unique solution*:
  - The basis selection $[S]$ determines one specific solution
  - BP soft decisions guide toward low-weight solutions
  - Matrix inversion on $H_([S])$ eliminates ambiguity
]

== Higher-Order OSD

OSD-0 assumes $bold(e)_([T]) = bold(0)$. This may miss the minimum-weight solution.

#definition[
  *Higher-order OSD* considers non-zero configurations of $bold(e)_([T])$.

  For any choice of $bold(e)_([T])$, the corresponding basis solution is:
  $ bold(e)_([S]) = H_([S])^(-1) dot (bold(s) + H_([T]) dot bold(e)_([T])) $

  This always satisfies $H dot bold(e) = bold(s)$ (verify by substitution).
]

*Challenge:* With $k' = n - r$ remainder bits, there are $2^(k')$ possible configurations — exhaustive search is infeasible!

== Combination Sweep Strategy (OSD-CS)

#definition[
  *Combination sweep* is a greedy search testing configurations by likelihood:

  #enum(
    [*Sort remainder bits:* Order bits in $[T]$ by BP soft decisions (most likely first)],
    [*Test weight-1:* Set each single bit in $bold(e)_([T])$ to 1 (all $k'$ possibilities)],
    [*Test weight-2:* Set each pair among the first $lambda$ bits to 1]
  )

  Keep the minimum-weight solution found.
]

#notation[
  The *binomial coefficient* $binom(lambda, 2) = (lambda(lambda-1))/2$ counts ways to choose 2 items from $lambda$.
]

Total configurations: $k' + binom(lambda, 2)$

With $lambda = 60$: $k' + 1770$ configurations (vs $2^(k')$ for exhaustive search!)

#pagebreak()

= The Complete BP+OSD Decoder

== Algorithm Flow

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Input box
    rect((-1.5, 6), (1.5, 7), radius: 0.1, name: "input")
    content("input", [Input: $H$, $bold(s)$, $p$])

    // BP box
    rect((-1.5, 4), (1.5, 5), radius: 0.1, fill: rgb("#e8f4e8"), name: "bp")
    content("bp", [Run BP])

    // Decision diamond
    line((0, 3.5), (1.2, 2.5), (0, 1.5), (-1.2, 2.5), close: true)
    content((0, 2.5), text(size: 8pt)[Converged?])

    // OSD box
    rect((3, 1.7), (5.5, 2.8), radius: 0.1, fill: rgb("#fff4e8"), name: "osd")
    content("osd", [Run OSD])

    // Output boxes
    rect((-1.5, -0.5), (1.5, 0.5), radius: 0.1, fill: rgb("#e8e8f4"), name: "out1")
    content("out1", [Return $bold(e)^"BP"$])

    rect((3, -0.5), (5.5, 0.5), radius: 0.1, fill: rgb("#e8e8f4"), name: "out2")
    content("out2", [Return $bold(e)^"OSD"$])

    // Arrows
    line((0, 6), (0, 5.1), mark: (end: ">"))
    line((0, 4), (0, 3.6), mark: (end: ">"))
    line((0, 1.5), (0, 0.6), mark: (end: ">"))
    content((-0.5, 1), text(size: 8pt)[Yes])

    line((1.2, 2.5), (2.9, 2.5), mark: (end: ">"))
    content((2, 2.9), text(size: 8pt)[No])

    line((4.25, 1.6), (4.25, 0.6), mark: (end: ">"))
  }),
  caption: [BP+OSD decoder flowchart]
)

#keypoint[
  - If BP succeeds (converges): use BP result — fast!
  - If BP fails: use OSD to resolve degeneracy — always gives valid answer
]

#pagebreak()

== Complete Algorithm: BP+OSD-CS

#figure(
  align(left)[
    #box(
      width: 100%,
      stroke: 1pt,
      inset: 12pt,
      radius: 4pt,
      fill: luma(250),
      [
        #text(weight: "bold", size: 10pt)[Algorithm 3: BP+OSD-CS Decoder]
        #v(0.5em)
        #text(size: 8.5pt)[
          ```
          Input: Parity matrix H (m×n, rank r), syndrome s, error prob p, depth λ=60
          Output: Error estimate e satisfying H·e = s

          function BP_OSD_CS(H, s, p, λ):
              // ===== STAGE 1: Run Belief Propagation (Algorithm 1) =====
              (converged, e_BP, P_1) = BP(H, s, p)
              if converged:
                  return e_BP

              // ===== STAGE 2: OSD-0 (Algorithm 2) =====
              [O_BP] = argsort(P_1)              // Sort: most likely flipped first
              H_sorted = H[:, O_BP]              // Reorder columns
              [S] = first r linearly independent columns of H_sorted
              [T] = remaining k' = n - r columns

              e_[S] = H_[S]^(-1) × s             // Solve on basis
              e_[T] = zeros(k')                  // Set remainder to zero
              best = (e_[S], e_[T])
              best_wt = hamming_weight(best)

              // ===== STAGE 3: Combination Sweep =====
              // Weight-1 search: try flipping each remainder bit
              for i = 0 to k'-1:
                  e_[T] = zeros(k');  e_[T][i] = 1
                  e_[S] = H_[S]^(-1) × (s + H_[T] × e_[T])
                  if hamming_weight((e_[S], e_[T])) < best_wt:
                      best = (e_[S], e_[T])
                      best_wt = hamming_weight(best)

              // Weight-2 search: try flipping pairs in first λ bits
              for i = 0 to min(λ, k')-1:
                  for j = i+1 to min(λ, k')-1:
                      e_[T] = zeros(k');  e_[T][i] = 1;  e_[T][j] = 1
                      e_[S] = H_[S]^(-1) × (s + H_[T] × e_[T])
                      if hamming_weight((e_[S], e_[T])) < best_wt:
                          best = (e_[S], e_[T])
                          best_wt = hamming_weight(best)

              return inverse_permute(best, O_BP)  // Remap to original ordering
          ```
        ]
      ]
    )
  ],
  caption: [Complete BP+OSD-CS algorithm]
)

#pagebreak()

= Results and Performance

== Error Threshold

#definition[
  The *threshold* $p_"th"$ is the maximum error rate below which the logical error rate decreases with increasing code distance.

  - If $p < p_"th"$: Larger codes $arrow.r$ exponentially better protection
  - If $p > p_"th"$: Larger codes $arrow.r$ worse protection (error correction fails)
]

== Experimental Results

#figure(
  table(
    columns: 4,
    align: center,
    stroke: 0.5pt,
    [*Code Family*], [*BP Only*], [*BP+OSD-0*], [*BP+OSD-CS*],
    [Toric], [N/A (fails)], [$9.2 plus.minus 0.2%$], [$bold(9.9 plus.minus 0.2%)$],
    [Semi-topological], [N/A (fails)], [$9.1 plus.minus 0.2%$], [$bold(9.7 plus.minus 0.2%)$],
    [Random QLDPC], [$6.5 plus.minus 0.1%$], [$6.7 plus.minus 0.1%$], [$bold(7.1 plus.minus 0.1%)$],
  ),
  caption: [Observed thresholds from the paper]
)

#box(
  width: 100%,
  stroke: 1pt + green,
  inset: 12pt,
  radius: 4pt,
  fill: rgb("#f5fff5"),
  [
    #text(weight: "bold")[Key Results for Toric Code]

    - *BP alone:* Complete failure due to degeneracy (no threshold)
    - *BP+OSD-CS:* 9.9% threshold (optimal decoder achieves 10.3%)
    - *Improvement:* Combination sweep gains ~0.7% over OSD-0
    - *Low-error regime:* Exponential suppression of logical errors
  ]
)

== Complexity

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    stroke: 0.5pt,
    [*Component*], [*Complexity*], [*Notes*],
    [BP (per iteration)], [$O(n)$], [Linear in block length],
    [OSD-0], [$O(n^3)$], [Dominated by matrix inversion],
    [Combination sweep], [$O(lambda^2)$], [$lambda = 60 arrow.r$ ~1830 trials],
    [*Total*], [$O(n^3)$], [Practical for moderate $n$],
  ),
  caption: [Complexity analysis]
)

#pagebreak()

= Summary

== Key Takeaways

#enum(
  [*Classical BP* computes marginal probabilities via message passing on factor graphs],

  [*Quantum codes suffer from degeneracy*: multiple errors can produce the same syndrome, causing BP to output invalid solutions (split beliefs)],

  [*OSD resolves degeneracy* by selecting a basis guided by BP soft decisions, then solving via matrix inversion to get a unique valid solution],

  [*Combination sweep* efficiently improves OSD-0 by testing low-weight configurations of the remainder bits],

  [*BP+OSD is general*: works for Toric codes, semi-topological codes, and random QLDPC codes, achieving near-optimal thresholds],
)

== The BP+OSD Recipe

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Box 1
    rect((-5, -0.8), (-2, 0.8), radius: 0.15, fill: rgb("#e8f4e8"), name: "b1")
    content((-3.5, 0.3), text(weight: "bold")[1. Run BP])
    content((-3.5, -0.3), text(size: 8pt)[Get $P_1$ values])

    // Box 2
    rect((-1, -0.8), (2, 0.8), radius: 0.15, fill: rgb("#fff4e8"), name: "b2")
    content((0.5, 0.3), text(weight: "bold")[2. Run OSD-0])
    content((0.5, -0.3), text(size: 8pt)[Use $P_1$ for basis])

    // Box 3
    rect((3, -0.8), (6, 0.8), radius: 0.15, fill: rgb("#e8e8f4"), name: "b3")
    content((4.5, 0.3), text(weight: "bold")[3. Sweep])
    content((4.5, -0.3), text(size: 8pt)[Try weight-1,2])

    // Arrows
    line((-1.9, 0), (-1.1, 0), mark: (end: ">"))
    line((2.1, 0), (2.9, 0), mark: (end: ">"))
  }),
  caption: [BP+OSD in three steps]
)

== Further Reading

- *Paper:* Roffe et al., Phys. Rev. Research 2, 043423 (2020), arXiv:2005.07016
- *Code:* #link("https://github.com/quantumgizmos/bp_osd")
- *Background:* MacKay, "Information Theory, Inference, and Learning Algorithms" (BP); Nielsen & Chuang, "Quantum Computation and Quantum Information" (QEC)

#v(2em)
#align(center)[#text(style: "italic")[End of Lecture Note]]
