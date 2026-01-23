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


#let keypoint(body) = block(
  width: 100%,
  stroke: 1pt + orange,
  inset: 10pt,
  radius: 4pt,
  fill: rgb("#fffaf0"),
  [#text(weight: "bold")[Key Point.] #body]
)

#let theorem(title, body) = block(
  width: 100%,
  stroke: (left: 3pt + purple),
  inset: (left: 12pt, y: 8pt),
  fill: rgb("#f8f0ff"),
  [#text(weight: "bold")[Theorem (#title).] #body]
)

#let proof(body) = block(
  width: 100%,
  inset: (left: 12pt, y: 8pt),
  [#text(weight: "bold", style: "italic")[Proof.] #body #h(1fr) $square$]
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

= Classical Error Correction

== Linear Codes

All arithmetic in this note is performed in *binary* (modulo 2):
- $0 + 0 = 0$, $quad$ $1 + 0 = 0 + 1 = 1$, $quad$ $1 + 1 = 0$
- This is also written as XOR: $a plus.o b = (a + b) mod 2$
- Vectors and matrices use element-wise mod-2 arithmetic

== Hamming Weight and Distance

#definition[
  The *Hamming weight* of a binary vector $bold(v)$ is the number of 1s it contains $|bold(v)| = sum_i v_i$.
  The *Hamming distance* between two vectors $bold(u)$ and $bold(v)$ is the number of positions where they differ:$d(bold(u), bold(v)) = |bold(u) + bold(v)|$.
]
For example, for $bold(v) = (1, 0, 1, 1, 0)$: Hamming weight $|bold(v)| = 3$
  and for $bold(u) = (1, 1, 0, 1, 0)$ and $bold(v) = (1, 0, 1, 1, 0)$: $bold(u) + bold(v) = (0, 1, 1, 0, 0)$ and $d(bold(u), bold(v)) = 2$.
#definition[
  An *$[n, k, d]$ linear code* $cal(C)$ is a set of binary vectors (called *codewords*) where $n$ is the *block length* (number of bits in each codeword), $k$ is the *dimension* (number of information bits encoded), and $d$ is the *minimum distance* (minimum Hamming weight among non-zero codewords). The *rate* of the code is $R = k\/n$.
  ]


A linear code can be defined by an $m times n$ *parity check matrix* $H$. $H_(i j)$ denotes the entry in row $i$, column $j$ of matrix $H$. $m$ is the number of rows in $H$ (number of parity checks), $n$ is the number of columns in $H$ (number of bits), and $"rank"(H)$ is the number of linearly independent rows. By the rank-nullity theorem: $k = n - "rank"(H)$.

For example, 
  The *$[3, 1, 3]$ repetition code* encodes 1 bit into 3 bits by triplication.

  Parity check matrix:
  $ H = mat(1, 1, 0; 0, 1, 1) $

  Verification: $H dot mat(0;0;0) = mat(0;0)$ ✓ and $H dot mat(1;1;1) = mat(0;0)$ ✓

  So the codewords are: $cal(C)_H = {(0,0,0), (1,1,1)}$

  Parameters: $n = 3$ bits, $k = 3 - 2 = 1$ info bit, $d = 3$ (weight of $(1,1,1)$)


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

== Probabilistic Graphical Models

Before introducing the Belief Propagation algorithm, we need to understand how probabilistic inference problems can be represented as graphs.

#definition[
  A *probabilistic graphical model* (PGM) is a graph-based representation of a probability distribution. Nodes represent random variables, and edges encode conditional independence relationships. PGMs enable efficient inference algorithms by exploiting the structure of the distribution.
]

There are two main families of PGMs:
- *Directed graphical models* (Bayesian networks): edges have direction, representing causal relationships
- *Undirected graphical models* (Markov networks): edges are undirected, representing symmetric dependencies

For error correction, we use undirected models because parity constraints are symmetric — no variable "causes" another.

== Undirected Probabilistic Graphical Models

#definition[
  An *undirected probabilistic graphical model* (also called a *Markov network* or *Markov random field*) represents a joint probability distribution as:
  $ P(bold(x)) = 1/Z product_(c in cal(C)) psi_c (bold(x)_c) $

  where:
  - $bold(x) = (x_1, ..., x_n)$ are random variables
  - $cal(C)$ is a set of *cliques* (fully connected subgraphs)
  - $psi_c (bold(x)_c)$ is a *potential function* over variables in clique $c$
  - $Z = sum_(bold(x)) product_c psi_c (bold(x)_c)$ is the *partition function* (normalization constant)
]

#keypoint[
  The UAI format mentioned in the getting started guide represents exactly this structure: variables (detectors), cliques (error mechanisms), and potential functions (error probabilities).
]

For binary error correction with syndrome $bold(s)$, we want to compute:
$ P(bold(e) | bold(s)) prop product_c psi_c (bold(e)_c) $

where each potential $psi_c$ encodes a parity constraint.

== Factor Graphs

To understand the Belief Propagation algorithm, we need the concept of *factor graphs*.

#definition[
  A *factor graph* is a bipartite graph $G = (V, U, E)$ representing the parity check matrix $H$.
The *data nodes* are set $V = {v_1, v_2, ..., v_n}$
    such that each node $v_j$ corresponds to each column of $H$. 
    A *parity nodes* are set $U = {u_1, u_2, ..., u_m}$
    such that each node $u_i$ corresponds to each row of $H$.
    An *edges* $E = {(v_j, u_i) : H_(i j) = 1}$
     connects $v_j$ to $u_i$ exists if $H_(i j) = 1$.
]

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
    content((8, 1), text(size: 12pt)[Factor graph for $H = mat(1,1,0; 0,1,1)$])
  }),
  caption: [Factor graph for the $[3,1,3]$ repetition code with node conventions]
)

The *neighborhoods* of nodes are defined as: $V(u_i) = {v_j : H_(i j) = 1}$.

== Comparing Graph Representations

Three related graph representations appear in coding theory and probabilistic inference:

#definition[
  *Comparison of graph representations:*

  1. *Undirected Probabilistic Graphical Model (Markov Network)*:
     - Nodes = random variables
     - Edges = direct dependencies between variables
     - Cliques = groups of mutually dependent variables
     - Represents: $P(bold(x)) = 1/Z product_c psi_c (bold(x)_c)$

  2. *Factor Graph*:
     - Two types of nodes: variable nodes AND factor nodes
     - Bipartite structure: edges only between variables and factors
     - Explicitly represents factorization of the distribution
     - Represents: $P(bold(x)) = 1/Z product_i f_i (bold(x)_(N(i)))$

  3. *Tanner Graph*:
     - A special case of factor graph for error correction codes
     - Variable nodes = bits (columns of $H$)
     - Factor nodes = parity checks (rows of $H$)
     - Represents: parity check matrix $H$ structure
]

#keypoint[
  *Key relationships:*
  - Factor graphs are a *bipartite refinement* of Markov networks that make the factorization explicit
  - Tanner graphs are factor graphs *specialized for linear codes* where factors represent parity constraints
  - All three represent the same probability distribution, but factor graphs enable more efficient message-passing algorithms
  - The UAI format represents Markov networks (cliques and potentials), while BP operates on the factor graph representation
]

#figure(
  table(
    columns: 4,
    align: center,
    stroke: 0.5pt,
    [*Property*], [*Markov Network*], [*Factor Graph*], [*Tanner Graph*],
    [Node types], [Variables only], [Variables + Factors], [Bits + Checks],
    [Graph structure], [General], [Bipartite], [Bipartite],
    [Edges represent], [Dependencies], [Factor membership], [Parity constraints],
    [Used for], [General inference], [Message passing], [Code decoding],
    [BP efficiency], [Less efficient], [Efficient], [Efficient],
  ),
  caption: [Comparison of graph representations]
)

*Why use factor graphs for BP?*
The bipartite structure of factor graphs makes message passing natural:
- Messages flow between variables and factors
- Each factor collects evidence from its variables
- Each variable aggregates information from its factors
- No need to handle complex clique structures

For error correction, the Tanner graph (factor graph) representation is ideal because:
- Parity checks are naturally factors (XOR constraints)
- Bits are naturally variables (error indicators)
- The sparse structure ($H$ has few 1s) gives efficient $O(n)$ BP iterations

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

= Belief Propagation Decoder

== Introduction and Motivation

The rediscovery of Low-Density Parity-Check (LDPC) codes in the late 1990s marked a paradigm shift in coding theory, transitioning from algebraic decoding algorithms to probabilistic iterative decoding that approaches the Shannon limit @mackay2003information. Central to this revolution is the *Belief Propagation* (BP) algorithm @pearl1988probabilistic, a message-passing protocol that operates on the graphical representation of codes.

#keypoint[
  *BP in Modern Communications:* BP decoding powers critical communication standards:
  - Wi-Fi (IEEE 802.11n/ac/ax)
  - Satellite communication (DVB-S2)
  - 5G New Radio

  While its practical efficacy is undisputed, the mathematical rigor underlying its convergence behavior involves multiple theoretical frameworks: Density Evolution for asymptotic analysis, Bethe Free Energy for variational optimization, and trapping set theory for failure mechanisms.
]

The convergence of BP is understood through different lenses depending on the regime. In the asymptotic limit of infinite block length, convergence is probabilistic and governed by Density Evolution @richardson2001capacity. In finite-length regimes, convergence is variational, linked to minimization of the Bethe Free Energy @yedidia2003understanding. However, combinatorial substructures known as trapping sets can arrest decoding, creating error floors @dolecek2010analysis.

== The Message Passing Mechanism

*Belief Propagation* (BP), also called the *sum-product algorithm*, is an iterative message-passing algorithm on the factor graph.

#definition[
  The goal of BP is to compute, for each bit $j$, the *marginal probability*:
  $ P_1(e_j) = P(e_j = 1 | bold(s)), $
  given $bold(s) = H dot bold(e)$ is the syndrome of the error $bold(e)$.
  This is called a *soft decision* -- it tells us how likely each bit is to be flipped.
]

We use the following notation throughout:
- $p$ as the channel error probability (probability each bit flips)
- $m_(v_j arrow.r u_i)$ as the message from data node $v_j$ to parity node $u_i$
- $m_(u_i arrow.r v_j)$ as the message from parity node $u_i$ to data node $v_j$
- Messages represent beliefs about whether $e_j = 1$

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
The BP-algorithm requires the quantification of how certain we are about the bit being flipped or not. This is done using *log-likelihood ratios* (LLR).
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

*Why?* Before any message passing, the only information we have about each bit is from the *channel itself*. Since each bit flips independently with probability $p$, the initial belief is simply the channel's prior: "this bit is probably correct" (because $p < 0.5$, so $p_l > 0$).

*Code:* The implementation adds small epsilon values for numerical stability:
```python
# Initialize channel LLRs
channel_llr = torch.log((1 - channel_probs + 1e-10) / (channel_probs + 1e-10))

# Initialize qubit-to-check messages with channel prior
msg_q2c = channel_llr[qubit_edges].unsqueeze(0).expand(batch_size, -1).clone()
msg_c2q = torch.zeros(batch_size, num_edges, device=device)
```

#v(0.5em)

#step-box(2, "Parity-to-Data Messages")[
  Each parity node $u_i$ sends a message to each connected data node $v_j$:
  - *Min-sum form:*
   $ m_(u_i arrow.r v_j) = (-1)^(s_i) dot alpha dot product_(v'_j in V(u_i) backslash v_j) "sign"(m_(v'_j arrow.r u_i)) dot min_(v'_j in V(u_i) backslash v_j) |m_(v'_j arrow.r u_i)| $ 

  - *Sum-Product form:*
  $ m_(u_i arrow.r v_j) = (-1)^(s_i) dot 2 tanh^(-1) lr(( product_(v'_j in V(u_i) backslash v_j) tanh lr(( m_(v'_j arrow.r u_i) / 2 )) )) $ 

  Where:
  - $s_i$ = the $i$-th syndrome bit (given as input, either 0 or 1)
  - $V(u_i) backslash v_j$ = all neighbors of $u_i$ except $v_j$ (defined in Section 3.5)
  - $"sign"(x) = +1$ if $x >= 0$, else $-1$
  - $alpha = 1 - 2^(-t)$ is a *damping factor* at iteration $t$ (helps convergence)
]

*Why?* A parity check enforces that XOR of all connected bits equals the syndrome bit $s_i$. The check node tells $v_j$: "Based on what I know about the *other* bits, here's how likely *you* are to be flipped."
If $s_i = 0$, the parity check says "even number of flipped bits." If the other bits all look correct (positive LLR), then $v_j$ should also be correct. If one other bit looks flipped (negative LLR), then $v_j$ should be correct to maintain even parity. The formula computes this XOR-like logic in LLR form.

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Parity node (center)
    rect((2.7, 0.7), (3.3, 1.3), name: "ui", fill: rgb("#ffe0e0"))
    content("ui", $u_i$)

    // Data nodes (tree-like above)
    circle((0, 3), radius: 0.35, name: "v1", fill: rgb("#e0ffe0"))
    content("v1", $v_1$)
    circle((2, 3), radius: 0.35, name: "v2", fill: rgb("#e0ffe0"))
    content("v2", $v_2$)
    circle((4, 3), radius: 0.35, name: "vj", fill: rgb("#e0e0ff"))
    content("vj", $v_j$)
    circle((6, 3), radius: 0.35, name: "v4", fill: rgb("#e0ffe0"))
    content("v4", $v_4$)

    // Incoming messages (green arrows)
    line((0, 2.6), (2.8, 1.4), mark: (end: ">"), stroke: 1.5pt + green)
    line((2, 2.6), (2.9, 1.4), mark: (end: ">"), stroke: 1.5pt + green)
    line((6, 2.6), (3.2, 1.4), mark: (end: ">"), stroke: 1.5pt + green)

    // Outgoing message to v_j (blue arrow, thicker)
    line((3.1, 1.4), (4, 2.6), mark: (end: ">"), stroke: 2pt + blue)
    content((4.8, 2), text(fill: blue, size: 9pt)[$m_(u_i arrow.r v_j)$])

    // Excluded message (dashed, gray)
    line((4, 2.6), (3.1, 1.4), stroke: (dash: "dashed", paint: gray))
    content((2.3, 2.2), text(fill: gray, size: 8pt)[excluded])

    // Labels
    content((3, -0.2), text(size: 8pt)[Check node collects info from $v_1, v_2, v_4$])
    content((3, -0.7), text(size: 8pt)[to compute message to $v_j$ (excluding $v_j$'s own message)])
  }),
  caption: [Parity-to-data message: $u_i$ uses info from all neighbors *except* $v_j$ to tell $v_j$ what it should be]
)

*Code:* The implementation computes signs and magnitudes separately, using a sorting trick to efficiently find the minimum excluding each edge:

```python
def _check_to_qubit_minsum(self, msg_q2c, syndromes):
    for c in range(num_checks):
        edges = self.check_to_edges[c]
        incoming = msg_q2c[:, edges]  # (batch, degree)

        # Separate signs and magnitudes
        signs = torch.sign(incoming)  # (batch, degree)
        mags = torch.abs(incoming)    # (batch, degree)

        # Product of all signs
        total_sign = torch.prod(signs, dim=1, keepdim=True)

        # Apply syndrome: flip sign if syndrome is 1
        syndrome_sign = 1 - 2 * syndromes[:, c:c+1]
        total_sign = total_sign * syndrome_sign

        # For each edge, divide out its sign to get product of others
        outgoing_signs = total_sign / (signs + 1e-10)

        # Min magnitude excluding each edge (second minimum trick)
        sorted_mags, _ = torch.sort(mags, dim=1)
        min_mag = sorted_mags[:, 0:1]
        second_min = sorted_mags[:, 1:2] if sorted_mags.shape[1] > 1 else min_mag

        # If edge has the min, use second_min; else use min
        is_min = (mags == min_mag)
        outgoing_mags = torch.where(is_min, second_min, min_mag)

        # Apply scaling factor
        scaling = 0.625
        msg_c2q[:, edges] = scaling * outgoing_signs * outgoing_mags
```

#keypoint[
  *Second Minimum Trick:* To compute $min_(v'_j eq.not v_j) |m_(v'_j)|$ efficiently, we sort all magnitudes once. For each edge: if it holds the minimum, use the second minimum; otherwise use the first minimum. This avoids recomputing $O(d)$ minimums for each of $d$ edges.
]

*Sum-Product Code:* The sum-product variant uses the tanh identity for exact computation:

```python
def _check_to_qubit_sumproduct(self, msg_q2c, syndromes):
    for c in range(num_checks):
        edges = self.check_to_edges[c]
        incoming = msg_q2c[:, edges]  # (batch, degree)

        # Compute tanh(LLR/2) - clamp for numerical stability
        half_llr = torch.clamp(incoming / 2, min=-20, max=20)
        tanh_vals = torch.tanh(half_llr)  # (batch, degree)

        # Product of all tanh values
        total_prod = torch.prod(tanh_vals, dim=1, keepdim=True)

        # Apply syndrome: flip sign if syndrome is 1
        syndrome_sign = 1 - 2 * syndromes[:, c:c+1]
        total_prod = total_prod * syndrome_sign

        # For each edge, divide out its tanh contribution
        outgoing_prod = total_prod / (tanh_vals + 1e-10)

        # Clamp to valid range for atanh (-1, 1)
        outgoing_prod = torch.clamp(outgoing_prod, min=-1+1e-7, max=1-1e-7)

        # Convert back: 2 * atanh(prod)
        msg_c2q[:, edges] = 2 * torch.atanh(outgoing_prod)
```

#keypoint[
  *Min-Sum vs Sum-Product:* Min-sum is an *approximation* of sum-product, not an equivalent algorithm. The relationship comes from the identity:
  $ 2 tanh^(-1) lr(( product_i tanh(x_i \/ 2) )) approx "sign" lr(( product_i x_i )) dot min_i |x_i| $
  
  This approximation holds because $tanh(x\/2) approx "sign"(x)$ for large $|x|$, and the product of tanh values is dominated by the smallest magnitude input. The scaling factor $alpha = 0.625$ in min-sum compensates for the systematic overestimation of confidence that results from this approximation @chen2005reduced.
]

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    stroke: 0.5pt,
    [*Algorithm*], [*Formula*], [*Trade-off*],
    [Sum-Product], [$2 tanh^(-1)(product tanh(x\/2))$], [Exact but slower (tanh/atanh)],
    [Min-Sum], [$alpha dot "sign"(product x) dot min|x|$], [Approximate but faster],
  ),
  caption: [Comparison of check-to-qubit message algorithms]
)

#v(0.5em)

#step-box(3, "Data-to-Parity Messages")[
  Each data node $v_j$ sends a message to each connected parity node $u_i$:

  $ m_(v_j arrow.r u_i) = p_l + sum_(u'_i in U(v_j) backslash u_i) m_(u'_i arrow.r v_j) $

  Where $U(v_j) backslash u_i$ = all parity neighbors of $v_j$ except $u_i$.
]

*Why?* A data node collects evidence from multiple parity checks. Each check provides independent information about whether this bit is flipped. The data node sums up all this evidence (in LLR, multiplication of probabilities becomes addition).

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Data node (center)
    circle((3, 2), radius: 0.4, name: "vj", fill: rgb("#e0e0ff"))
    content("vj", $v_j$)

    // Parity nodes (tree-like below)
    rect((-0.3, -0.3), (0.3, 0.3), name: "u1", fill: rgb("#ffe0e0"))
    content("u1", $u_1$)
    rect((2.7, -0.3), (3.3, 0.3), name: "u2", fill: rgb("#ffe0e0"))
    content("u2", $u_2$)
    rect((5.7, -0.3), (6.3, 0.3), name: "ui", fill: rgb("#fff0e0"))
    content("ui", $u_i$)

    // Incoming messages (green arrows)
    line((0.1, 0.4), (2.7, 1.7), mark: (end: ">"), stroke: 1.5pt + green)
    content((0.8, 1.3), text(fill: green, size: 8pt)[$m_(u_1 arrow.r v_j)$])
    line((3, 0.4), (3, 1.55), mark: (end: ">"), stroke: 1.5pt + green)
    content((3.9, 0.9), text(fill: green, size: 8pt)[$m_(u_2 arrow.r v_j)$])

    // Outgoing message to u_i (blue arrow)
    line((3.3, 1.7), (5.9, 0.4), mark: (end: ">"), stroke: 2pt + blue)
    content((5.5, 1.3), text(fill: blue, size: 8pt)[$m_(v_j arrow.r u_i)$])

    // Excluded message (dashed)
    line((5.9, 0.4), (3.3, 1.7), stroke: (dash: "dashed", paint: gray))
    content((5.8, 1.8), text(fill: gray, size: 8pt)[excluded])

    // Channel prior
    line((3, 3.5), (3, 2.45), mark: (end: ">"), stroke: 1.5pt + purple)
    content((3, 3.8), text(fill: purple, size: 8pt)[channel prior $p_l$])

    // Labels
    content((3, -1.2), text(size: 8pt)[Data node sums: channel prior + messages from $u_1, u_2$])
    content((3, -1.7), text(size: 8pt)[to send to $u_i$ (excluding $u_i$'s own message)])
  }),
  caption: [Data-to-parity message: $v_j$ combines channel prior with info from other checks]
)

*Intuition:* Why exclude $u_i$? To avoid *echo effects*. If $v_j$ included the message it previously received from $u_i$, that information would bounce back, creating a feedback loop. On a *tree-structured graph*, this exclusion ensures each piece of evidence is counted exactly once, making BP exact. On graphs with cycles, this is an approximation.

*Code:* The implementation efficiently computes this by computing the total sum and subtracting each edge's contribution:

```python
def _qubit_to_check(self, msg_c2q, channel_llr):
    for q in range(num_qubits):
        edges = self.qubit_to_edges[q]
        incoming = msg_c2q[:, edges]  # (batch, degree)

        # Sum of all incoming messages plus channel prior
        total_sum = incoming.sum(dim=1, keepdim=True) + channel_llr[q]

        # For each edge, subtract its own contribution
        msg_q2c[:, edges] = total_sum - incoming
```

#keypoint[
  *Why subtract?* Instead of computing $n-1$ sums for each of $n$ edges (expensive), we compute one total sum and subtract each term. This reduces complexity from $O(d^2)$ to $O(d)$ per qubit.
]


#step-box(4, "Compute Soft Decisions")[
  For each bit $j$, compute the total belief (sum of all evidence):

  $ P_1(e_j) = p_l + sum_(u_i in U(v_j)) m_(u_i arrow.r v_j) $
]

*Why?* Unlike Step 3, here we include *all* incoming messages (no exclusion). This is the final belief about bit $j$, combining the channel prior with evidence from *every* connected parity check. The result is the log-posterior probability ratio.

*Code:* The implementation sums all incoming messages (no exclusion) and converts to probability:

```python
def _compute_marginals(self, msg_c2q, channel_llr):
    for q in range(num_qubits):
        edges = self.qubit_to_edges[q]

        # Total LLR = channel + sum of ALL incoming
        total_llr = channel_llr[q] + msg_c2q[:, edges].sum(dim=1)

        # Convert LLR to probability: P(1) = sigmoid(-LLR)
        marginals[:, q] = torch.sigmoid(-total_llr)
```

#keypoint[
  *Sigmoid conversion:* The relationship between LLR and probability is:
  $ "LLR" = log P(0)/P(1) arrow.r.double P(1) = 1/(1 + e^("LLR")) = sigma(-"LLR") $
  PyTorch's `torch.sigmoid(-total_llr)` computes this efficiently.
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

*Why?* The sign of LLR directly tells us the most likely value: $P_1 > 0$ means $P(e_j = 0) > P(e_j = 1)$, so the bit is probably correct. $P_1 < 0$ means the bit is probably flipped.

#v(0.5em)

#step-box(6, "Check Convergence")[
  Verify if the estimate satisfies the syndrome equation:

  $ H dot bold(e)^"BP" = bold(s) quad ? $

  - *If yes:* BP has *converged*. Return $bold(e)^"BP"$ and soft decisions $P_1$.
  - *If no:* Go back to Step 2 and repeat.
  - *If max iterations reached:* BP has *failed to converge*.
]

*Why iterate?* On graphs with cycles, a single pass doesn't propagate information globally. Each iteration allows beliefs to travel further through the graph. Eventually, if the error is correctable, the hard decisions will satisfy all parity checks.

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Iteration 1
    content((1, 2.5), text(weight: "bold", size: 9pt)[Iteration 1])
    circle((0, 1), radius: 0.25, fill: rgb("#e0ffe0"))
    circle((1, 1), radius: 0.25, fill: rgb("#ffe0e0"))
    circle((2, 1), radius: 0.25, fill: rgb("#e0e0e0"))
    line((0.3, 1), (0.7, 1), mark: (end: ">"), stroke: blue)
    content((1, 0.3), text(size: 7pt)[local info])

    // Iteration 2
    content((4.5, 2.5), text(weight: "bold", size: 9pt)[Iteration 2])
    circle((3.5, 1), radius: 0.25, fill: rgb("#e0ffe0"))
    circle((4.5, 1), radius: 0.25, fill: rgb("#d0ffd0"))
    circle((5.5, 1), radius: 0.25, fill: rgb("#ffe0e0"))
    line((3.8, 1), (4.2, 1), mark: (end: ">"), stroke: blue)
    line((4.8, 1), (5.2, 1), mark: (end: ">"), stroke: blue)
    content((4.5, 0.3), text(size: 7pt)[info spreads])

    // Iteration N
    content((8, 2.5), text(weight: "bold", size: 9pt)[Iteration N])
    circle((7, 1), radius: 0.25, fill: rgb("#c0ffc0"))
    circle((8, 1), radius: 0.25, fill: rgb("#c0ffc0"))
    circle((9, 1), radius: 0.25, fill: rgb("#c0ffc0"))
    line((7.3, 1), (7.7, 1), mark: (end: ">"), stroke: green)
    line((8.3, 1), (8.7, 1), mark: (end: ">"), stroke: green)
    content((8, 0.3), text(size: 7pt)[global consensus])
  }),
  caption: [Information propagates further with each iteration until convergence]
)

=== Damping for Convergence

*Theory:* Damping prevents oscillation by mixing old and new messages:
$ m^((t+1)) = gamma dot m^((t)) + (1 - gamma) dot m_"new" $

where $gamma in [0, 1)$ is the damping factor.

*Code:* Applied after computing new check-to-qubit messages:

```python
# Main iteration loop
for _ in range(max_iter):
    # Compute new check-to-qubit messages
    msg_c2q_new = self._check_to_qubit_minsum(msg_q2c, syndromes)

    # Damping: blend old and new messages
    msg_c2q = damping * msg_c2q + (1 - damping) * msg_c2q_new

    # Update qubit-to-check messages
    msg_q2c = self._qubit_to_check(msg_c2q, channel_llr)
```

#figure(
  table(
    columns: 3,
    align: (center, left, left),
    stroke: 0.5pt,
    [*Damping Value*], [*Behavior*], [*Use Case*],
    [$gamma = 0$], [No damping (full update)], [Simple graphs, fast convergence],
    [$gamma = 0.2$], [Light damping], [Typical default],
    [$gamma = 0.5$], [Strong damping], [Graphs with short cycles],
  ),
  caption: [Effect of damping factor on BP convergence]
)

=== Summary: Complete BP Iteration

Putting it all together, one BP iteration consists of:

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Step boxes
    rect((-5, 1.5), (-2.5, 2.5), radius: 0.1, fill: rgb("#e8f4e8"), name: "s1")
    content("s1", text(size: 9pt)[Check→Qubit])

    rect((-1.5, 1.5), (1, 2.5), radius: 0.1, fill: rgb("#fff4e8"), name: "s2")
    content("s2", text(size: 9pt)[Damping])

    rect((2, 1.5), (4.5, 2.5), radius: 0.1, fill: rgb("#e8e8f4"), name: "s3")
    content("s3", text(size: 9pt)[Qubit→Check])

    // Arrows
    line((-2.4, 2), (-1.6, 2), mark: (end: ">"))
    line((1.1, 2), (1.9, 2), mark: (end: ">"))

    // Formula annotations
    content((-3.75, 0.8), text(size: 8pt)[$m_(u arrow.r v) = "minsum/sumproduct"$])
    content((-0.25, 0.8), text(size: 8pt)[$m' = gamma m + (1-gamma) m_"new"$])
    content((3.25, 0.8), text(size: 8pt)[$m_(v arrow.r u) = p_l + sum m - m_"in"$])
  }),
  caption: [One BP iteration: message updates with damping]
)

#pagebreak()

== BP Dynamics: Classical vs. Quantum Constraints

To understand why BP fails on quantum codes, we compare its behavior on a minimal classical graph (2 bits) versus a minimal quantum stabilizer (4 bits).

=== Problem Setup: 2-Bit vs. 4-Bit Parity

Consider two scenarios with channel error probability $p=0.1$ and observed syndrome $s=1$ (odd parity): (i). A check node $u_1$ connected to 2 variables $v_1, v_2$ (e.g., a simple parity check).
(ii). A check node $u_1$ connected to 4 variables $v_1, ..., v_4$ (e.g., a surface code plaquette).

#figure(
  grid(
    columns: 2,
    gutter: 1cm,
    canvas(length: 1cm, {
      import draw: *
      // 2-bit graph
      circle((-1, 1.5), radius: 0.3, fill: rgb("#e0ffe0"), name: "v1")
      content("v1", $v_1$)
      circle((1, 1.5), radius: 0.3, fill: rgb("#e0ffe0"), name: "v2")
      content("v2", $v_2$)
      rect((-0.3, -0.3), (0.3, 0.3), fill: rgb("#ffe0e0"), name: "u1")
      content("u1", $u_1$)
      line("v1", "u1", stroke: blue)
      line("v2", "u1", stroke: blue)
      content((0, -1.1), text(size: 9pt)[*Case A: 2-Bit Check*])
    }),
    canvas(length: 1cm, {
      import draw: *
      // 4-bit graph
      circle((-1, 1), radius: 0.3, fill: rgb("#e0ffe0"), name: "v1")
      content("v1", $v_1$)
      circle((1, 1), radius: 0.3, fill: rgb("#e0ffe0"), name: "v2")
      content("v2", $v_2$)
      circle((1, -1), radius: 0.3, fill: rgb("#e0ffe0"), name: "v3")
      content("v3", $v_3$)
      circle((-1, -1), radius: 0.3, fill: rgb("#e0ffe0"), name: "v4")
      content("v4", $v_4$)
      rect((-0.3, -0.3), (0.3, 0.3), fill: rgb("#ffe0e0"), name: "u1")
      content("u1", $u_1$)
      line("v1", "u1", stroke: blue)
      line("v2", "u1", stroke: blue)
      line("v3", "u1", stroke: blue)
      line("v4", "u1", stroke: blue)
      content((0, -1.6), text(size: 9pt)[*Case B: 4-Bit Plaquette*])
    })
  ),
  caption: [Comparison of message passing geometry]
)

Both cases start with the same channel LLR:
$ "LLR"_"channel" = ln((1-p)/p) = ln(9) approx 2.197. $
The check node sends a message to $v_1$ based on evidence from *all other* neighbors.
Formula: $m_(u arrow.r v) = (-1)^s dot 2 tanh^(-1)( product_(v' in N(u) without v) tanh(m_(v' arrow.r u) \/ 2) )$

*Case A (2-Bit): Strong Correction*
$v_1$ has only 1 neighbor ($v_2$). The product contains a single term $tanh(1.1) approx 0.8$.
$ m_(u_1 arrow.r v_1) = -2 tanh^(-1)(0.8) approx -2.197 $
The check says: "My other neighbor is definitely correct, so *you* must be wrong." The message magnitude *matches* the channel prior but flips the sign.

*Case B (4-Bit): Signal Dilution*
$v_1$ has 3 neighbors ($v_2, v_3, v_4$). The product accumulates uncertainty: $0.8^3 approx 0.512$.
$ m_(u_1 arrow.r v_1) = -2 tanh^(-1)(0.512) approx -1.13 $
The check says: "I see an error, but it could be any of you 4. I suspect you, but weakly." The penalty (-1.13) is *weaker* than the channel prior (+2.197).

=== Step 3: Variable Update and Failure

We now compute the updated belief for $v_1$:

#table(
  columns: 3,
  align: center,
  stroke: none,
  table.header([*Metric*], [*Case A (2-Bit)*], [*Case B (4-Bit)*]),
  table.hline(),
  [Prior LLR], [$+2.197$], [$+2.197$],
  [Check Msg], [$-2.197$], [$-1.13$],
  [*Net LLR*], [$bold(0)$], [$bold(+1.067)$],
  [Prob($e_1=1$)], [$50%$], [$approx 26%$],
  [Hard Decision], [$e_1=0$ or $1$], [$e_1=0$],
)

*Why Quantum BP Fails:*
1.  *Case A (Classical):* The LLR drops to 0. BP correctly identifies "maximum uncertainty." It knows it cannot distinguish between $e_1$ and $e_2$.
2.  *Case B (Quantum):* The LLR remains positive ($+1.067$). BP remains "confident" that the bit is correct.
    - The hard decision outputs $bold(e) = (0,0,0,0)$.
    - *Parity Check:* $0+0+0+0 = 0 != 1$.
    - *Result:* BP fails to find a valid solution because the "blame" is diluted across too many qubits.

In a full surface code, this effect compounds. Multiple conflicting checks reduce the LLR toward 0, leading to a state of "Split Belief" where the decoder is paralyzed by symmetry.

#keypoint[
  *Summary of the Failure Mode:*
  - *Ambiguity Dilution:* In high-weight stabilizers (like 4-bit plaquettes), the check node message is too weak to overturn the channel prior.
  - *Invalid Hard Decisions:* Unlike the 2-bit case where uncertainty is explicit ($L=0$), the 4-bit case results in false confidence ($L>0$) that violates the parity constraint.
]

=== The Degeneracy Problem

#box(
  width: 100%,
  stroke: 2pt + red,
  inset: 12pt,
  radius: 4pt,
  fill: rgb("#fff5f5"),
  [
    #text(weight: "bold", fill: red)[The Degeneracy Problem]
    In quantum codes, *multiple distinct error patterns* can be physically equivalent (differing only by a stabilizer).
    BP, which is designed to find a *unique* correct error, fails to distinguish between these equivalent patterns, leading to decoding failure.
  ]
)

#definition[
  Two errors $bold(e)_1$ and $bold(e)_2$ are *degenerate* if they produce the same syndrome $bold(s)$ and their difference forms a stabilizer:
  $ H dot bold(e)_1 = H dot bold(e)_2 = bold(s) quad "and" quad bold(e)_1 + bold(e)_2 in "Stabilizers" $
  
  Unlike classical codes where distinct low-weight errors are rare, quantum stabilizer codes are *constructed* from local degeneracies (the stabilizers themselves).
]

Fro example, consider the Toric code where qubits reside on the edges of a lattice. A stabilizer $B_p$ acts on the 4 qubits surrounding a plaquette.
The weight-2 error occurs on the *left* edges is degenerate with the weight-2 error on the *right* two edges.

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Define vertices of the plaquette
    let p1 = (-2, 2)
    let p2 = (2, 2)
    let p3 = (2, -2)
    let p4 = (-2, -2)

    // Draw the stabilizer plaquette background
    rect((-2, -2), (2, 2), fill: rgb("#f9f9f9"), stroke: (dash: "dashed"))
    content((0, 0), text(size: 10pt, fill: gray)[Stabilizer $B_p$])

    // Vertices (Checks)
    circle(p1, radius: 0.2, fill: black)
    circle(p2, radius: 0.2, fill: black)
    circle(p3, radius: 0.2, fill: black)
    circle(p4, radius: 0.2, fill: black)

    // Syndrome: Highlight the top-left and bottom-left vertices
    // Let's assume the string is open, creating defects at corners.
    // For a weight-2 error on a loop, the syndrome is technically 0 if it's a stabilizer.
    // Let's model a logical error scenario or an open string:
    // Case: Error on Left (e1) vs Error on Right (e2)
    // Both connect the top and bottom rows.
    
    // Path 1 (Left) - Red
    line(p1, p4, stroke: (paint: red, thickness: 3pt), name: "e1")
    content((-2.5, 0), text(fill: red, weight: "bold")[$e_L$])
    
    // Path 2 (Right) - Blue
    line(p2, p3, stroke: (paint: blue, thickness: 3pt), name: "e2")
    content((2.5, 0), text(fill: blue, weight: "bold")[$e_R$])

    // Labels explaining the symmetry
    content((0, -2.8), text(size: 9pt)[$e_L$ and $e_R$ are equivalent (differ by $B_p$)])
    content((0, -3.3), text(size: 9pt)[Both have weight 2. Both satisfy local checks.])
  }),
  caption: [Symmetric degeneracy in a Toric code plaquette]
)

When BP runs on this symmetric structure, it encounters a fatal ambiguity.

1.  *Perfect Symmetry:* The graph structure for $e_L$ is identical to the graph structure for $e_R$.
2.  *Message Stalling:* BP receives equal evidence for the left-side error and the right-side error.
3.  *Marginal Probability 0.5:* The algorithm converges to a state where every qubit in the loop has $P(e_i) approx 0.5$.

#definition[
  *The Hard Decision Failure:*
  When $P(e_i) approx 0.5$, the log-likelihood ratio is $approx 0$.
  
  - If we threshold *below* 0.5: The decoder outputs $bold(e) = bold(0)$ (no error). This fails to satisfy the syndrome.
  - If we threshold *above* 0.5: The decoder outputs $bold(e) = e_L + e_R$. This is the stabilizer itself (a closed loop), which effectively applies a logical identity but fails to correct the actual open string error.
]

This is why BP alone typically exhibits *zero threshold* on the Toric code: as the code size increases, the number of these symmetric loops increases, and BP gets confused by all of them simultaneously.

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

= Ordered Statistics Decoding (OSD)

== The Key Insight

The parity check matrix $H$ (size $m times n$ with $n > m$) has more columns than rows and cannot be directly inverted. However, we can select a subset of $r = "rank"(H)$ linearly independent columns to form an invertible $m times r$ submatrix.
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
    rect((1, -1), (3, 1), fill: rgb("#e0ffe0"), name: "HS")
    content("HS", $H_([S])$)
    content((1.75, -1.5), text(size: 9pt)[$m times r$])
    content((1.75, -2), text(size: 8pt)[invertible!])

    // Remainder submatrix
    rect((3.5, -1), (5, 1), fill: rgb("#ffe0e0"), name: "HT")
    content("HT", $H_([T])$)
    content((4, -1.5), text(size: 9pt)[$m times k'$])
  }),
  caption: [Splitting $H$ into basis and remainder parts]
)
  OSD then resolves split beliefs by *forcing a unique solution*:
  - The basis selection $[S]$ determines one specific solution
  - BP soft decisions guide toward low-weight solutions
  - Matrix inversion on $H_([S])$ eliminates ambiguity

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

The Python implementation in `osd.py` realizes OSD-0 through Gaussian elimination rather than explicit matrix inversion. Below is the mapping between the theoretical definition and the code logic.

*Step 1: Choosing a "Good" Basis (Sorting)*

The definition requires selecting basis columns $[S]$ based on the reliability of BP soft decisions. The code achieves this by calculating how far each probability is from maximum uncertainty ($0.5$).

```python
# 2. Sort (Soft Decision)
# Sort by reliability: |p - 0.5| descending (most reliable first)
reliability = np.abs(probs - 0.5)
sorted_indices = np.argsort(reliability)[::-1]

```

- `np.abs(probs - 0.5)`:  Quantifies certainty. If $P=0.99$, reliability is $0.49$ (High). If $P=0.51$, reliability is $0.01$ (Low).
- `argsort(...)[::-1]`: Creates a permutation where the most reliable bits are indices $0, 1, 2, ...$. These will be prioritized to form the basis $[S]$.
*Step 2: solving for basis bits (RREF)*

Instead of computing  explicitly, the code computes the *Reduced Row Echelon Form* (RREF) of the augmented matrix. This is numerically stable and solves the linear system in one pass.

```python
# 3. Build the augmented matrix [H_sorted | s] and compute RREF
augmented, pivot_cols = self._get_rref_cached(sorted_indices, syndrome)

```

- `augmented`: Represents the system after Gaussian elimination. The columns corresponding to the basis  are transformed into an identity matrix structure.
- `pivot_cols`: The indices of the first linearly independent columns found. These effectively define the basis set .

*Step 3: setting remainder to zero and extracting solution*

The code initializes the solution vector to zeros and then updates *only* the basis positions using the transformed syndrome.

```python
# Basis solution (OSD-0 Solution): Assume all free variables are 0
solution_base = np.zeros(self.num_errors, dtype=np.int8)

for r in range(augmented.shape[0]):
    # Find the pivot column in this row
    row_pivots = np.where(augmented[r, :self.num_errors] == 1)[0]
    if len(row_pivots) > 0:
        col = row_pivots[0] # The pivot column
        if col in pivot_cols:
            # OSD-0 assignment: pivot_bit = transformed_syndrome
            solution_base[col] = augmented[r, -1]

```

- `solution_base = np.zeros(...)`: Ensures that any bit not explicitly updated remains 0. This satisfies the OSD-0 constraint .
- `augmented[r, -1]`: This is the value of the syndrome  after row operations. Since the basis submatrix is now Identity-like, this value is the solution for the corresponding basis bit.

#figure(
table(
columns: 2,
align: (left, left),
stroke: 0.5pt,
[*Theoretical Step*], [*Code Realization*],
[1. Choose Basis ], [`argsort(|probs - 0.5|)` prioritizes reliable bits],
[2. Matrix Inversion], [`_compute_rref` diagonalizes the basis submatrix],
[3. Solve for ], [`solution_base[col] = augmented[r, -1]`],
[4. Set ], [`np.zeros(...)` initialization],
),
caption: [Mapping OSD-0 theory to Python implementation]
)


== Higher-Order OSD

OSD-0 assumes the remainder error bits are zero ($bold(e)_([T]) = bold(0)$). While this provides a valid solution, it forces all "correction" work onto the basis bits $[S]$, which may result in a high-weight (improbable) error pattern.

#definition[
  *Higher-order OSD* improves this by testing non-zero configurations for the remainder bits $bold(e)_([T])$.

  For any chosen hypothesis $bold(e)_([T])$, the corresponding basis bits $bold(e)_([S])$ are uniquely determined to satisfy the syndrome:
  $ bold(e)_([S]) = H_([S])^(-1) dot (bold(s) + H_([T]) dot bold(e)_([T])) $
]

=== Verification by Substitution

It is trivial to show that the constructed error $bold(e) = (bold(e)_([S]), bold(e)_([T]))$ always satisfies the parity check equation $H dot bold(e) = bold(s)$.

$ H dot bold(e) &= mat(H_([S]), H_([T])) dot mat(bold(e)_([S]); bold(e)_([T])) \
  &= H_([S]) dot bold(e)_([S]) + H_([T]) dot bold(e)_([T]) \
  &= H_([S]) dot [H_([S])^(-1) dot (bold(s) + H_([T]) dot bold(e)_([T]))] + H_([T]) dot bold(e)_([T]) \
  &= I dot (bold(s) + H_([T]) dot bold(e)_([T])) + H_([T]) dot bold(e)_([T]) \
  &= bold(s) + H_([T]) dot bold(e)_([T]) + H_([T]) dot bold(e)_([T]) \
  &= bold(s) + bold(0) = bold(s) $

(Note: In binary arithmetic, $x + x = 0$.) [cite_start][cite: 264]

=== The Search Challenge

The remainder set $[T]$ has size $k' = n - r$.
- *Exhaustive Search (OSD-E):* Testing all $2^(k')$ patterns guarantees finding the minimum weight solution but has exponential complexity.
- *Search Depth ($lambda$):* To make this feasible, we restrict the search to the $lambda$ "least reliable" bits in $[T]$ (those with BP probabilities closest to 0.5).
- *Problem:* Even with a restricted depth $lambda$, checking all $2^lambda$ patterns is too slow if we want $lambda$ to be large (e.g., $lambda > 20$).

== Combination Sweep Strategy (OSD-CS)

To allow for a larger search depth (e.g., $lambda approx 50-100$) without exponential cost, we use the *Combination Sweep* strategy.

#definition[
  *OSD-CS* assumes the true error pattern on the remainder bits is *sparse*. Instead of checking all $2^lambda$ patterns, it only checks patterns with low Hamming weight.

  *Algorithm Steps:*
  1. *Sort:* Select the $lambda$ least reliable positions in $[T]$.
  2. *Sweep:* Generate candidate vectors $bold(e)_([T])$ with:
     - *Weight 0:* The zero vector (equivalent to OSD-0).
     - *Weight 1:* All single-bit flips among the $lambda$ bits.
     - *Weight 2:* All pairs of bit flips among the $lambda$ bits.
  3. *Select:* Calculate $bold(e)_([S])$ for each candidate, compute the total weight, and pick the best one.
]

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    stroke: 0.5pt,
    [*Method*], [*Complexity*], [*Use Case*],
    [OSD-0], [$O(1)$], [Fastest, baseline performance],
    [OSD-E (Exhaustive)], [$O(2^lambda)$], [Optimal for small $lambda$ ($\le 15$)],
    [OSD-CS (Comb. Sweep)], [$O(lambda^2)$], [Near-optimal for large $lambda$ ($ approx 60$)],
  ),
  caption: [Comparison of OSD Search Strategies]
)

#keypoint[
  *Why OSD-CS works for Quantum Codes:*
  In the low-error regime relevant for QEC, it is statistically very unlikely that the optimal solution requires flipping 3+ bits in the specific subset of "uncertain" remainder bits.
  
  [cite_start]Checking only weights 0, 1, and 2 captures the vast majority of likely error configurations while reducing complexity from exponential to polynomial (quadratic). [cite: 266]
]

#definition[
  *Combination sweep* is a greedy search testing configurations by likelihood:

  #enum(
    [*Sort remainder bits:* Order bits in $[T]$ by BP soft decisions (most likely first)],
    [*Test weight-1:* Set each single bit in $bold(e)_([T])$ to 1 (all $k'$ possibilities)],
    [*Test weight-2:* Set each pair among the first $lambda$ bits to 1]
  )

  Keep the minimum-weight solution found.
]

Recall that the *binomial coefficient* $binom(lambda, 2) = (lambda(lambda-1))/2$ counts ways to choose 2 items from $lambda$.

Total configurations: $k' + binom(lambda, 2)$

With $lambda = 60$: $k' + 1770$ configurations (vs $2^(k')$ for exhaustive search!)


=== OSD-CS Implementation Analysis

The Python implementation realizes the Combination Sweep (OSD-CS) strategy by explicitly generating sparse error patterns (weights 0, 1, and 2) instead of iterating through all binary combinations. This logic is encapsulated in the `_generate_osd_cs_candidates` method.

*Step 1: Generating Sparse Candidates*

The code generates a list of candidate vectors $bold(e)_([T])$ for the remainder bits. Unlike OSD-E which uses bit-shifting to generate $2^k$ integers, OSD-CS uses structured loops to generate $O(lambda^2)$ specific patterns.

```python
def _generate_osd_cs_candidates(self, k: int, osd_order: int) -> List[np.ndarray]:
    candidates = []

    # 1. Weight 0: Zero vector (all free variables = 0)
    # Corresponds to OSD-0 solution
    candidates.append(np.zeros(k, dtype=np.int8))

    # 2. Weight 1: Single-bit flips
    # Corresponds to trying one flipped bit in the search region
    for i in range(k):
        candidate = np.zeros(k, dtype=np.int8)
        candidate[i] = 1
        candidates.append(candidate)

    # 3. Weight 2: Two-bit flips
    # Corresponds to trying pairs of flipped bits
    # The search is limited by 'osd_order' (lambda)
    limit = min(osd_order, k)
    for i in range(limit):
        for j in range(i + 1, limit):
            candidate = np.zeros(k, dtype=np.int8)
            candidate[i] = 1
            candidate[j] = 1
            candidates.append(candidate)

    return candidates

```

- `np.zeros(k)`: Adds the "baseline" hypothesis that the remainder bits are error-free.


- `range(k)` loop: Adds  candidates, each representing a single bit flip at index `i`.


- `range(limit)` nested loop: Adds roughly  candidates, representing pairs of flips at indices `(i, j)`.



*Step 2: Integration into the Solve Loop*

In the `solve` method, the code detects the `osd_method` flag and switches the candidate generation strategy. The rest of the solving logic (calculating syndrome, solving for basis bits, scoring) remains identical to OSD-E.

```python
# Generate candidates based on method
if osd_method == 'combination_sweep':
    # OSD-CS: Generate single and double bit flip candidates
    # len(search_cols) is the number of free variables being searched
    candidates = self._generate_osd_cs_candidates(len(search_cols), osd_order)
else:
    # Exhaustive: Generate all 2^k combinations
    # ... bit shifting logic ...

```

*Complexity Comparison in Code*

*OSD-E:* The list `candidates` has length $2^k$ (where $k$ is `len(search_cols)`). The `for e_T_search in candidates` loop runs exponentially many times.


*OSD-CS:* The list `candidates` has length . The loop runs quadratically many times, allowing  to be much larger.



#figure(
table(
columns: 2,
align: (left, left),
stroke: 0.5pt,
[*Weight Class*], [*Code Realization*],
[Weight 0], [`candidates.append(np.zeros(k))` ],
[Weight 1], [`for i in range(k): candidate[i] = 1` ],
[Weight 2], [`for i... for j...: candidate[i]=1; candidate[j]=1` ],
),
caption: [Mapping OSD-CS theory to Python loops]
)

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


== BP Convergence and Performance Guarantees

#theorem("BP Convergence on Trees")[@pearl1988probabilistic @montanari2008belief
  If the factor graph $G = (V, U, E)$ is a *tree* (contains no cycles), then BP converges to the *exact* marginal probabilities $P(e_j = 1 | bold(s))$ in at most $d$ iterations, where $d$ is the diameter of the tree (maximum distance between any two nodes).
]

#proof[
  We prove exactness by induction on the tree structure, using the factorization property of graphical models.

  *Factorization on Trees:* For a tree-structured factor graph, the joint probability distribution factors as:
  $ P(bold(x)) = 1/Z product_(a in cal(F)) psi_a(bold(x)_(cal(N)(a))) $
  where $cal(F)$ is the set of factors, $cal(N)(a)$ are neighbors of factor $a$, and $Z$ is the partition function.

  *Key Property:* On a tree, removing any node $v$ separates the graph into disjoint connected components (subtrees). By the global Markov property, variables in different subtrees are conditionally independent given $v$.

  *Base Case (Leaf Nodes):* Consider a leaf variable node $v$ with single neighbor (factor) $a$. The message $mu_(v arrow.r a)(x_v)$ depends only on the local evidence $P(y_v | x_v)$. Since there are no other dependencies, this message is exact at iteration 1.

  *Inductive Step:* Assume messages from all nodes at distance $> k$ from root are exact. Consider node $u$ at distance $k$ with neighbors $cal(N)(u) = {a_1, ..., a_m}$.

  For message $mu_(u arrow.r a_i)(x_u)$, the BP update is:
  $ mu_(u arrow.r a_i)(x_u) prop P(y_u | x_u) product_(a_j in cal(N)(u) without {a_i}) mu_(a_j arrow.r u)(x_u) $

  By the separation property, removing $u$ creates $m$ independent subtrees rooted at ${a_1, ..., a_m}$. By the inductive hypothesis, messages from these subtrees are exact marginals of their respective subtrees. Since subtrees are conditionally independent given $u$, the product of messages equals the joint probability of all subtree configurations, making $mu_(u arrow.r a_i)(x_u)$ exact.

  *Termination:* After $d$ iterations (where $d$ is the tree diameter), messages have propagated from all leaves to all nodes. Each node's belief $b_v(x_v) prop P(y_v | x_v) product_(a in cal(N)(v)) mu_(a arrow.r v)(x_v)$ equals the exact marginal $P(x_v | bold(y))$ by the factorization property.

  Therefore, BP computes exact marginals on trees in $d$ iterations.
]

*Example:* Consider the $[7, 4, 3]$ Hamming code with tree-structured factor graph:

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Data nodes (top)
    for (i, x) in ((0, 0), (1, 2), (2, 4), (3, 6)) {
      circle((x, 3), radius: 0.3, name: "v" + str(i), fill: rgb("#e0ffe0"))
      content("v" + str(i), $v_#i$)
    }

    // Parity nodes (bottom)
    for (i, x) in ((0, 1), (1, 3), (2, 5)) {
      rect((x - 0.3, -0.3), (x + 0.3, 0.3), name: "u" + str(i), fill: rgb("#ffe0e0"))
      content("u" + str(i), $u_#i$)
    }

    // Tree edges (no cycles)
    line((0, 2.7), (0.8, 0.3))    // v0-u0
    line((2, 2.7), (1.2, 0.3))    // v1-u0
    line((2, 2.7), (2.8, 0.3))    // v1-u1
    line((4, 2.7), (3.2, 0.3))    // v2-u1
    line((4, 2.7), (4.8, 0.3))    // v2-u2
    line((6, 2.7), (5.2, 0.3))    // v3-u2

    content((3, -1.2), text(size: 9pt)[Tree structure: diameter $d = 4$, BP converges in 4 iterations])
  }),
  caption: [Tree-structured code where BP gives exact solution]
)

For this tree with syndrome $bold(s) = (1, 0, 0)$ and $p = 0.1$:
- BP converges in $d = 4$ iterations
- Output: $bold(e)^"BP" = (1, 0, 0, 0, 0, 0, 0)$ (single bit flip at position 0)
- This is the *exact* maximum likelihood solution

#v(1em)

#theorem("BP Performance on Graphs with Cycles")[@richardson2008modern @tatikonda2002loopy
  For an $(l, q)$-LDPC code with factor graph of *girth* $g$ (minimum cycle length), BP provides the following guarantees:

  1. *Local optimality:* If the true error $bold(e)^*$ has Hamming weight $|bold(e)^*| < g\/2$, then BP converges to $bold(e)^*$ with high probability (for sufficiently small $p$).

  2. *Approximation bound:* For codes with girth $g >= 6$ and maximum degree $Delta = max(l, q)$, if BP converges, the output $bold(e)^"BP"$ satisfies:
  $ |bold(e)^"BP"| <= (1 + epsilon(g, Delta)) dot |bold(e)^*| $
  where $epsilon(g, Delta) arrow.r 0$ as $g arrow.r infinity$ for fixed $Delta$.

  3. *Iteration complexity:* BP requires $O(g)$ iterations to propagate information across the shortest cycle.
]

#proof[
  *Part 1 (Local optimality):* Consider an error $bold(e)^*$ with $|bold(e)^*| < g\/2$. In the factor graph, the neighborhood of radius $r = floor(g\/2) - 1$ around any error bit is a tree (no cycles within distance $r$). Within this tree neighborhood:
  - BP computes exact marginals (by Theorem 1)
  - The error bits are separated by distance $>= g\/2$
  - No interference between error regions

  Therefore, BP correctly identifies each error bit independently, giving $bold(e)^"BP" = bold(e)^*$.

  *Part 2 (Approximation bound):* For $|bold(e)^*| >= g\/2$, cycles create dependencies. The approximation error comes from:
  - *Double-counting:* Evidence circulates through cycles
  - *Correlation:* Nearby error bits are not independent

  For girth $g$, the correlation decays exponentially with distance. The number of length-$g$ cycles through a node is bounded by $Delta^g$. Using the correlation decay lemma for loopy belief propagation, the relative error in log-likelihood ratios is:
  $ epsilon(g, Delta) <= C dot Delta^(2-g\/2) $
  for some constant $C$. This translates to the weight approximation bound.

  *Part 3 (Iteration complexity):* Information propagates one edge per iteration. To detect a cycle of length $g$, messages must travel distance $g$, requiring $O(g)$ iterations.
]

#keypoint[
  *Practical implications:*
  - Codes with large girth $g$ (e.g., $g >= 8$) allow BP to correct more errors
  - Random LDPC codes typically have $g = O(log n)$, giving good BP performance
  - Structured codes (e.g., Toric code with $g = 4$) have small girth, leading to BP failures
  - The degeneracy problem in quantum codes compounds the cycle problem, making OSD necessary
]

=== Density Evolution Framework

We now develop the rigorous theoretical foundations that explain *when* and *why* BP converges in different regimes, drawing from asymptotic analysis via density evolution @richardson2001capacity, variational optimization through statistical physics @yedidia2003understanding, and combinatorial failure modes @dolecek2010analysis.

For infinite-length random LDPC codes, convergence is analyzed through the *density evolution* method @richardson2008modern, which tracks the probability distributions of messages rather than individual message values.

#definition[
  *Cycle-Free Horizon:* For a random LDPC code with block length $n arrow.r infinity$, the *computation tree* of depth $l$ rooted at any edge is the subgraph containing all nodes reachable within $l$ hops. The cycle-free horizon property states:
  $ lim_(n arrow.r infinity) bb(P)(text("cycle in depth-")l text(" tree")) = 0 $

  This means that for any fixed number of iterations $l$, the local neighborhood appears tree-like with probability approaching 1 as $n arrow.r infinity$.
]

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Root edge
    circle((0, 0), radius: 0.15, fill: blue.lighten(60%))
    content((0, 0), text(size: 8pt, fill: blue)[root])

    // Depth 1
    for i in range(3) {
      let x = (i - 1) * 1.5
      circle((x, -1.2), radius: 0.12, fill: red.lighten(70%))
      line((0, -0.15), (x, -1.08))
    }

    // Depth 2
    for i in range(3) {
      for j in range(2) {
        let x = (i - 1) * 1.5 + (j - 0.5) * 0.6
        let y = -2.2
        circle((x, y), radius: 0.1, fill: blue.lighten(60%))
        line(((i - 1) * 1.5, -1.32), (x, y + 0.1))
      }
    }

    content((0, -3), text(size: 9pt)[Computation tree of depth $l=2$: no cycles])
    content((3.5, -1), text(size: 8pt, fill: red)[check nodes])
    content((3.5, -1.5), text(size: 8pt, fill: blue)[variable nodes])
  }),
  caption: [Locally tree-like structure in large random graphs]
)

#keypoint[
  The cycle-free horizon is the mathematical justification for applying tree-based convergence proofs to loopy graphs in the asymptotic limit. It explains why BP performs well on long random LDPC codes despite the presence of cycles.
]

#definition[
  *Concentration Theorem:* Let $Z$ be a performance metric (e.g., bit error rate) of BP after $l$ iterations on a code randomly drawn from ensemble $cal(C)(n, lambda, rho)$, where $lambda(x)$ and $rho(x)$ are the variable and check node degree distributions. For any $epsilon > 0$:
  $ bb(P)(|Z - bb(E)[Z]| > epsilon) <= e^(-beta n epsilon^2) $
  where $beta > 0$ depends on the ensemble parameters.

  *Interpretation:* As $n arrow.r infinity$, almost all codes in the ensemble perform identically to the ensemble average. Individual code performance concentrates around the mean with exponentially small deviation probability.
]

#keypoint[
  *Concentration visualization:* The performance metric $Z$ concentrates exponentially around its ensemble average $bb(E)[Z]$. For large block length $n$, the probability of deviation greater than $epsilon$ decays as $e^(-beta n epsilon^2)$, meaning almost all codes perform identically to the average.
]

The proof of the Concentration Theorem uses martingale theory:

#proof[
  *Proof sketch via Doob's Martingale:*

  1. *Martingale Construction:* View code selection as revealing edges sequentially. Define $Z_i = bb(E)[Z | text("first ") i text(" edges revealed")]$. This forms a Doob martingale: $bb(E)[Z_(i+1) | Z_0, ..., Z_i] = Z_i$.

  2. *Bounded Differences:* In a sparse graph with maximum degree $Delta$, changing a single edge affects at most $O(Delta^l)$ messages after $l$ iterations. Since $Delta$ is constant and $l$ is fixed, the change in $Z$ is bounded: $|Z_i - Z_(i-1)| <= c\/n$ for some constant $c$.

  3. *Azuma-Hoeffding Inequality:* For a martingale with bounded differences $|Z_i - Z_(i-1)| <= c_i$:
  $ bb(P)(|Z_m - Z_0| > epsilon) <= 2 exp(-(epsilon^2)/(2 sum_(i=1)^m c_i^2)) $

  4. *Application:* With $m = O(n)$ edges and $c_i = O(1\/n)$, we have $sum c_i^2 = O(1\/n)$, giving:
  $ bb(P)(|Z - bb(E)[Z]| > epsilon) <= 2 exp(-(epsilon^2 n)/(2 C)) = e^(-beta n epsilon^2) $
  where $beta = 1\/(2C)$.
]

#theorem("Threshold Theorem")[
  For a code ensemble with degree distributions $lambda(x), rho(x)$ and a symmetric channel with noise parameter $sigma$ (e.g., standard deviation for AWGN), there exists a unique *threshold* $sigma^*$ such that:

  1. If $sigma < sigma^*$ (low noise): As $l arrow.r infinity$, the probability of decoding error $P_e^((l)) arrow.r 0$

  2. If $sigma > sigma^*$ (high noise): $P_e^((l))$ remains bounded away from zero

  The threshold is determined by the fixed points of the density evolution recursion:
  $ P_(l+1) = Phi(P_l, sigma) $
  where $Phi$ is the density update operator combining variable and check node operations.
]

#keypoint[
  *Threshold phenomenon:* There exists a sharp transition at $sigma^*$. Below this threshold (low noise), BP converges to zero error as iterations increase. Above threshold (high noise), errors persist. This sharp phase transition is characteristic of random LDPC ensembles.
]

#keypoint[
  *Why the threshold exists:* The density evolution operator $Phi$ has two competing fixed points:
  - *All-correct fixed point:* Messages concentrate at $plus.minus infinity$ (high confidence)
  - *Error fixed point:* Messages remain near zero (low confidence)

  Below threshold, the all-correct fixed point is stable and attracts all trajectories. Above threshold, the error fixed point becomes stable, trapping the decoder.
]

=== Variational Perspective: Bethe Free Energy

The density evolution framework applies to infinite-length codes. For finite loopy graphs, we need a different lens: *statistical physics* @yedidia2003understanding. This reveals that BP is actually performing *variational optimization* of an energy function.

#definition[
  *Bethe Free Energy:* For a factor graph with variables $bold(x) = (x_1, ..., x_n)$ and factors $psi_a$, let $b_i(x_i)$ be the *belief* (pseudo-marginal) at variable $i$ and $b_a(bold(x)_a)$ be the belief at factor $a$. The Bethe Free Energy is:
  $ F_"Bethe"(b) = sum_a sum_(bold(x)_a) b_a(bold(x)_a) E_a(bold(x)_a) - H_"Bethe"(b) $

  where the *Bethe entropy* approximates the true entropy using local entropies:
  $ H_"Bethe" = sum_a H(b_a) + sum_i (1 - d_i) H(b_i) $

  Here $d_i$ is the degree of variable $i$, and $H(b) = -sum_x b(x) log b(x)$ is the Shannon entropy.

  *Constraints:* Beliefs must be normalized and *marginally consistent*:
  $ sum_(bold(x)_a without x_i) b_a(bold(x)_a) = b_i(x_i) quad "for all" i in a $
]

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Simple 3-node factor graph
    circle((-1.5, 0), radius: 0.15, fill: blue.lighten(60%))
    content((-1.5, 0), text(size: 8pt)[$x_1$])

    circle((1.5, 0), radius: 0.15, fill: blue.lighten(60%))
    content((1.5, 0), text(size: 8pt)[$x_2$])

    rect((-0.15, -0.15), (0.15, 0.15), fill: red.lighten(70%))
    content((0, 0), text(size: 8pt)[$psi$])

    line((-1.35, 0), (-0.15, 0), stroke: gray)
    line((0.15, 0), (1.35, 0), stroke: gray)

    // Entropy terms
    content((-1.5, -0.8), text(size: 8pt, fill: blue)[$H(b_1)$])
    content((1.5, -0.8), text(size: 8pt, fill: blue)[$H(b_2)$])
    content((0, -0.8), text(size: 8pt, fill: red)[$H(b_psi)$])

    content((0, -1.5), text(size: 9pt)[Bethe entropy: $H_"Bethe" = H(b_psi) + (1-2)H(b_1) + (1-2)H(b_2)$])
    content((0, -2), text(size: 8pt)[$(d_1 = d_2 = 2$ for this graph$)$])
  }),
  caption: [Bethe entropy decomposes global entropy into local terms]
)

#keypoint[
  *Intuition:* The Bethe approximation treats each factor independently, summing local entropies. The $(1 - d_i)$ correction prevents double-counting: a variable connected to $d_i$ factors appears in $d_i$ factor entropies, so we subtract $(d_i - 1)$ copies of its individual entropy.
]

#theorem("Yedidia-Freeman-Weiss")[@yedidia2003understanding
  A set of beliefs $\\{b_i, b_a\\}$ is a *fixed point* of the Sum-Product BP algorithm if and only if it is a *stationary point* (critical point) of the Bethe Free Energy $F_"Bethe"(b)$ subject to normalization and marginalization constraints.

  *Equivalently:* BP performs coordinate descent on the Bethe Free Energy. Each message update corresponds to minimizing $F_"Bethe"$ with respect to one edge's belief.
]

#proof[
  *Proof sketch via Lagrangian:*

  1. *Constrained optimization:* Form the Lagrangian:
  $ cal(L) = F_"Bethe"(b) + sum_(i,a) sum_(x_i) lambda_(i a)(x_i) (b_i(x_i) - sum_(bold(x)_a without x_i) b_a(bold(x)_a)) + "normalization terms" $

  2. *Stationarity conditions:* Taking $partial cal(L) \/ partial b_a = 0$ and $partial cal(L) \/ partial b_i = 0$:
  $ b_a(bold(x)_a) prop psi_a(bold(x)_a) product_(i in a) exp(lambda_(i a)(x_i)) $
  $ b_i(x_i) prop product_(a in i) exp(lambda_(a i)(x_i)) $

  3. *Message identification:* Define messages $mu_(i arrow.r a)(x_i) = exp(lambda_(i a)(x_i))$. Substituting and enforcing marginalization constraints yields exactly the BP update equations:
  $ mu_(i arrow.r a)(x_i) prop P(y_i | x_i) product_(a' in i without a) mu_(a' arrow.r i)(x_i) $
  $ mu_(a arrow.r i)(x_i) prop sum_(bold(x)_a without x_i) psi_a(bold(x)_a) product_(i' in a without i) mu_(i' arrow.r a)(x_(i')) $
]

#keypoint[
  *Energy landscape interpretation:* BP performs gradient descent on the Bethe Free Energy landscape. On trees, there's a single global minimum (correct solution). On loopy graphs, local minima can trap the decoder, corresponding to incorrect fixed points. The contour lines represent energy levels, with BP trajectories flowing toward minima.
]

#keypoint[
  *Implications for convergence:*
  - *On trees:* Bethe approximation is exact ($F_"Bethe" = F_"Gibbs"$), so BP finds the global minimum
  - *On loopy graphs:* $F_"Bethe"$ is an approximation. BP finds a local minimum, which may not be the true posterior
  - *Stable fixed points* correspond to local minima of $F_"Bethe"$
  - *Unstable fixed points* (saddle points) cause oscillations

  This explains why BP can converge to incorrect solutions: it gets trapped in local minima created by graph cycles.
]

=== Sufficient Conditions for Convergence

While density evolution guarantees asymptotic convergence and Bethe theory explains fixed points, neither provides *guarantees* for specific finite loopy graphs. We now present rigorous sufficient conditions @ihler2005loopy.

#definition[
  *Dobrushin's Influence Matrix:* For a graphical model, the influence $C_(i j)$ measures the maximum change in the marginal distribution of variable $i$ caused by fixing variable $j$:
  $ C_(i j) = sup_(x_j, x_j') ||P(x_i | x_j) - P(x_i | x_j')||_"TV" $

  where $|| dot ||_"TV"$ is the total variation distance.

  The *Dobrushin interdependence matrix* $bold(C)$ has entries $C_(i j)$ for $i eq.not j$ and $C_(i i) = 0$.
]

#theorem("Dobrushin's Uniqueness Condition")[
  If the Dobrushin matrix satisfies:
  $ ||bold(C)||_infinity = max_i sum_(j eq.not i) C_(i j) < 1 $

  then:
  1. The Gibbs measure has a unique fixed point
  2. BP converges exponentially fast to this fixed point from any initialization
  3. The convergence rate is $lambda = ||bold(C)||_infinity$
]

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Small graph example
    for i in range(4) {
      let angle = i * 90deg
      let x = 1.5 * calc.cos(angle)
      let y = 1.5 * calc.sin(angle)
      circle((x, y), radius: 0.15, fill: blue.lighten(60%))
      content((x, y), text(size: 8pt)[$x_#(i+1)$])
    }

    // Edges
    for i in range(4) {
      let angle1 = i * 90deg
      let angle2 = calc.rem(i + 1, 4) * 90deg
      let x1 = 1.5 * calc.cos(angle1)
      let y1 = 1.5 * calc.sin(angle1)
      let x2 = 1.5 * calc.cos(angle2)
      let y2 = 1.5 * calc.sin(angle2)
      line((x1, y1), (x2, y2), stroke: gray)
    }

    // Influence matrix
    content((0, -2.5), text(size: 9pt)[Example: 4-cycle with weak coupling])
    content((0, -3), text(size: 8pt)[$||bold(C)||_infinity = max_i sum_j C_(i j) = 2 dot 0.3 = 0.6 < 1$ ✓])
  }),
  caption: [Dobrushin condition: information dissipates through the graph]
)

#keypoint[
  *Limitation for LDPC codes:* Error correction codes are designed to *propagate* information over long distances. Parity checks impose hard constraints (infinite coupling strength). Therefore, useful LDPC codes typically *violate* Dobrushin's condition.

  While sufficient, Dobrushin's condition is far from necessary. It applies mainly to high-noise regimes where correlations are weak.
]

#theorem("Contraction Mapping Convergence")[
  View BP as a mapping $bold(T): cal(M) arrow.r cal(M)$ on the space of messages. If $bold(T)$ is a *contraction* under some metric $d$:
  $ d(bold(T)(bold(m)), bold(T)(bold(m'))) <= lambda dot d(bold(m), bold(m')) $
  with Lipschitz constant $lambda < 1$, then:

  1. BP has a unique fixed point $bold(m)^*$
  2. BP converges geometrically: $d(bold(m)^((t)), bold(m)^*) <= lambda^t d(bold(m)^((0)), bold(m)^*)$
]

#proof[
  *Proof:* Direct application of the Banach Fixed Point Theorem. The contraction property ensures:
  - *Uniqueness:* If $bold(m)^*$ and $bold(m')^*$ are both fixed points, then:
  $ d(bold(m)^*, bold(m')^*) = d(bold(T)(bold(m)^*), bold(T)(bold(m')^*)) <= lambda dot d(bold(m)^*, bold(m')^*) $
  Since $lambda < 1$, this implies $d(bold(m)^*, bold(m')^*) = 0$, so $bold(m)^* = bold(m')^*$.

  - *Convergence:* For any initialization $bold(m)^((0))$:
  $ d(bold(m)^((t+1)), bold(m)^*) = d(bold(T)(bold(m)^((t))), bold(T)(bold(m)^*)) <= lambda dot d(bold(m)^((t)), bold(m)^*) $
  Iterating gives $d(bold(m)^((t)), bold(m)^*) <= lambda^t d(bold(m)^((0)), bold(m)^*)$.
]

#keypoint[
  *Spectral radius condition:* For binary pairwise models, the contraction constant can be computed from the *spectral radius* of the interaction matrix:
  $ rho(bold(A)) < 1, quad "where" A_(i j) = tanh |J_(i j)| $

  This is sharper than Dobrushin's condition (which corresponds to the $L_infinity$ norm of $bold(A)$).
]

=== Failure Mechanisms: Trapping Sets

The previous sections explain when BP converges. We now characterize when and why it *fails* @dolecek2010analysis. In the high-SNR regime, BP can get trapped in incorrect fixed points due to specific graph substructures.

#definition[
  *(a,b) Absorbing Set:* A subset $cal(D) subset.eq V$ of $a$ variable nodes is an $(a, b)$ absorbing set if:

  1. The induced subgraph contains exactly $b$ *odd-degree* check nodes (unsatisfied checks)
  2. Every variable node $v in cal(D)$ has *strictly more* even-degree neighbors than odd-degree neighbors in the induced subgraph

  *Interpretation:* If the variables in $cal(D)$ are in error, each receives more "confirming" messages (from satisfied checks) than "correcting" messages (from unsatisfied checks), causing the decoder to stabilize in the error state.
]

#figure(
  canvas(length: 1cm, {
    import draw: *

    // The canonical (5,3) absorbing set
    // 5 variable nodes in a specific configuration
    let var_pos = (
      (-1.5, 0),
      (-0.75, 1),
      (0.75, 1),
      (1.5, 0),
      (0, -0.5)
    )

    // Variable nodes
    for (i, pos) in var_pos.enumerate() {
      circle(pos, radius: 0.15, fill: red.lighten(40%))
      content(pos, text(size: 7pt, fill: white, weight: "bold")[$v_#(i+1)$])
    }

    // Check nodes (3 odd-degree, others even-degree)
    let check_pos = (
      (-1.1, 0.5),   // odd
      (0, 0.7),      // odd
      (1.1, 0.5),    // odd
      (-0.5, -0.8),  // even (degree 2)
      (0.5, -0.8)    // even (degree 2)
    )

    for (i, pos) in check_pos.enumerate() {
      let color = if i < 3 { rgb("#ff8800") } else { rgb("#00cc00") }
      rect((pos.at(0) - 0.12, pos.at(1) - 0.12), (pos.at(0) + 0.12, pos.at(1) + 0.12),
           fill: color.lighten(60%))
      content(pos, text(size: 6pt)[$c_#(i+1)$])
    }

    // Edges (simplified connectivity)
    line(var_pos.at(0), check_pos.at(0))
    line(var_pos.at(1), check_pos.at(0))
    line(var_pos.at(1), check_pos.at(1))
    line(var_pos.at(2), check_pos.at(1))
    line(var_pos.at(2), check_pos.at(2))
    line(var_pos.at(3), check_pos.at(2))
    line(var_pos.at(4), check_pos.at(3))
    line(var_pos.at(0), check_pos.at(3))
    line(var_pos.at(4), check_pos.at(4))
    line(var_pos.at(3), check_pos.at(4))

    content((0, -1.8), text(size: 9pt, fill: red)[5 error variables (red)])
    content((0, -2.2), text(size: 9pt, fill: rgb("#ff8800"))[3 odd-degree checks (orange)])
    content((0, -2.6), text(size: 9pt, fill: rgb("#00cc00"))[Even-degree checks (green)])
    content((0, -3.2), text(size: 8pt)[Canonical $(5,3)$ absorbing set: each variable has $>=2$ even neighbors])
  }),
  caption: [The $(5,3)$ absorbing set: a stable error configuration]
)

#keypoint[
  *Why BP gets trapped:*
  1. *Majority vote:* Each variable node performs a weighted majority vote of its check neighbors
  2. *Satisfied checks dominate:* In an absorbing set, satisfied checks (even degree) outnumber unsatisfied checks (odd degree) for each variable
  3. *Reinforcement loop:* Satisfied checks send messages that *confirm* the error state, while unsatisfied checks send weak correction signals
  4. *Stable fixed point:* The configuration becomes a local minimum of the Bethe Free Energy

  This is the primary cause of *error floors* in LDPC codes: at high SNR, rare noise patterns that activate absorbing sets dominate the error probability.
]

#theorem("Absorbing Sets and Error Floors")[
  For an LDPC code with minimum absorbing set size $(a_"min", b_"min")$, the error floor is dominated by:
  $ P_"error" approx binom(n, a_"min") dot p^(a_"min") dot (1-p)^(n-a_"min") dot P_"trap" $

  where $P_"trap"$ is the probability that BP fails to correct the absorbing set configuration.

  *Implication:* Error floor height is determined by the *size* and *multiplicity* of small absorbing sets. Code design focuses on eliminating small absorbing sets.
]

#pagebreak()


= Minimum Weight Perfect Matching (MWPM) Decoder

== Maximum Likelihood Decoding and MWPM

Maximum Likelihood Decoding (MLD) seeks the most probable error pattern $bold(e)$ given syndrome $bold(s)$ and error probabilities $p(bold(e))$.

#definition[
  *Maximum Likelihood Decoding Problem:* Given parity check matrix $bold(H) in bb(F)_2^(m times n)$, syndrome $bold(s) in bb(F)_2^m$, and error weights $w_i = ln((1-p_i)/p_i)$, find:
  $ min_(bold(c) in bb(F)_2^n) sum_(i in [n]) w_i c_i quad "subject to" quad bold(H) bold(c) = bold(s) $
]

For certain code structures, MLD can be efficiently reduced to a graph matching problem.

#theorem("MLD to MWPM Reduction")[
  If every column of $bold(H)$ has at most 2 non-zero elements (each error triggers at most 2 detectors), then MLD can be deterministically reduced to Minimum Weight Perfect Matching with boundaries in polynomial time.
]

#definition[
  *Detector Graph:* Given $bold(H) in bb(F)_2^(m times n)$, construct graph $G = (V, E)$ where:
  - Vertices: $V = [m] union {0}$ (detectors plus boundary vertex)
  - Edges: For each column $i$ of $bold(H)$:
    - If column $i$ has weight 2 (triggers detectors $x_1, x_2$): edge $(x_1, x_2)$ with weight $w_i$
    - If column $i$ has weight 1 (triggers detector $x$): edge $(x, 0)$ with weight $w_i$
]

#proof[
  *Reduction procedure:*

  1. *Graph construction:* Build detector graph $G$ from $bold(H)$ as defined above. The boundary operator $partial: bb(F)_2^n arrow.r bb(F)_2^(m+1)$ maps edge vectors to vertex vectors, corresponding to the parity check matrix.

  2. *Syndrome to boundary:* Given syndrome $bold(s) in bb(F)_2^m$, identify the set $D subset.eq V$ of vertices with non-zero syndrome values. This becomes the boundary condition for the matching problem.

  3. *Shortest path computation:* For all pairs $(u, v)$ where $u, v in D union {0}$, compute shortest paths using Dijkstra's algorithm. This requires $O(|D|^2)$ shortest path computations, constructing a complete weighted graph on $D union {0}$.

  4. *MWPM with boundary:* Solve MWPM on the complete graph with boundary vertex ${0}$. The solution gives edges whose boundary equals $D$, which corresponds to the minimum weight error pattern satisfying $bold(H) bold(c) = bold(s)$.

  Since each step is polynomial time, MLD reduces to MWPM in polynomial time.
]

== The Matching Polytope

#definition[
  *Weighted Perfect Matching:* Given weighted graph $G = (V, E, W)$ where $W = {w_e in bb(R) | e in E}$:
  - A *matching* $M subset.eq E$ has no two edges sharing a vertex
  - A *perfect matching* covers every vertex in $V$
  - The *weight* of matching $M$ is $sum_(e in M) w_e$
]

The integer programming formulation uses indicator variables $x_e in {0, 1}$:

$ min_bold(x) sum_(e in E) w_e x_e quad "subject to" quad sum_(e in delta({v})) x_e = 1 space forall v in V, quad x_e in {0, 1} $

where $delta({v})$ denotes edges incident to vertex $v$.

#theorem("Matching Polytope Characterization")[
  Define the odd set family $cal(O)(G) = {U subset.eq V : |U| "is odd and" >= 3}$. Let:
  $ P_1(G) &= "conv"{bold(x) : x_e in {0,1}, sum_(e in delta({v})) x_e = 1 space forall v in V} \
  P_2(G) &= {bold(x) : x_e >= 0, sum_(e in delta({v})) x_e = 1 space forall v in V, sum_(e in delta(U)) x_e >= 1 space forall U in cal(O)(G)} $

  If edge weights are rational, then $P_1(G) = P_2(G)$.

  *Implication:* The integer program can be relaxed to a linear program by replacing $x_e in {0,1}$ with $x_e >= 0$ and adding the *blossom constraints* $sum_(e in delta(U)) x_e >= 1$ for all odd sets $U$.
]

#keypoint[
  *Why blossom constraints matter:* An odd set $U$ cannot have a perfect matching using only internal edges (odd number of vertices). Therefore, at least one edge must connect to the outside: $sum_(e in delta(U)) x_e >= 1$. This constraint is necessary and sufficient for the convex hull to equal the integer hull.
]

== Dual Formulation and Optimality Conditions

#definition[
  *MWPM Dual Problem:* The dual of the MWPM linear program is:
  $ max_(bold(y)) sum_(v in V) y_v + sum_(O in cal(O)(G)) y_O $
  subject to:
  $ lambda_e = w_e - (y_(v_1) + y_(v_2)) - sum_(O: e in delta(O)) y_O >= 0 quad forall e in E \
  y_O >= 0 quad forall O in cal(O)(G) $

  where $lambda_e$ is the *slack* of edge $e$.
]

#theorem("KKT Complementary Slackness")[
  Primal solution $bold(x)$ and dual solution $(bold(y), {y_O})$ are optimal if and only if:
  1. *Primal feasibility:* $sum_(e in delta({v})) x_e = 1$, $sum_(e in delta(U)) x_e >= 1$, $x_e >= 0$
  2. *Dual feasibility:* $lambda_e >= 0$, $y_O >= 0$
  3. *Complementary slackness:*
     - $lambda_e x_e = 0$ (tight edges are in matching)
     - $y_O (sum_(e in delta(O)) x_e - 1) = 0$ (tight odd sets have positive dual)
]

== The Blossom Algorithm

The Blossom algorithm, developed by Edmonds (1965), solves MWPM by maintaining primal and dual feasibility while growing alternating trees.

#definition[
  *Alternating structures:*
  - *M-alternating walk:* Path $(v_0, v_1, ..., v_t)$ where edges alternate between $M$ and $E without M$
  - *M-augmenting path:* M-alternating walk with both endpoints unmatched
  - *M-blossom:* Odd-length cycle in an M-alternating walk where edges alternate in/out of $M$
]

#keypoint[
  *Algorithm overview:*

  1. *Initialization:* Start with empty matching $M = emptyset$, dual variables $y_v = 0$

  2. *Main loop:* While $M$ is not perfect:
     - *Search:* Find M-alternating walks from unmatched vertices
     - *Augment:* If M-augmenting path found, flip edges along path (add unmatched edges, remove matched edges)
     - *Shrink:* If M-blossom found, contract it to a single vertex, update dual variables
     - *Grow:* If no path/blossom found, increase dual variables to make new edges tight, add to search tree
     - *Expand:* When blossom dual variable reaches zero, uncontract it

  3. *Termination:* When all vertices are matched, return $M$
]

#theorem("Blossom Algorithm Correctness and Complexity")[
  The Blossom algorithm:
  1. Maintains primal feasibility, dual feasibility, and complementary slackness throughout
  2. Terminates with an optimal MWPM
  3. Runs in $O(|V|^3)$ time with careful implementation

  *Iteration bounds:*
  - Augmentations: at most $|V|\/2$
  - Contractions: at most $2|V|$
  - Expansions: at most $2|V|$
  - Edge additions: at most $3|V|$
  - Total: $O(|V|^2)$ iterations, each taking $O(|V|)$ time
]

#keypoint[
  *Why MWPM for quantum codes:*
  - Surface codes and other topological codes have parity check matrices where each error triggers at most 2 stabilizers
  - This structure allows efficient MLD via MWPM
  - Blossom algorithm provides polynomial-time optimal decoding
  - Practical implementations achieve near-optimal thresholds (~10-11% for surface codes)
  - Contrast with BP: MWPM finds global optimum but is slower; BP is faster but can get trapped in local minima
]

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

Common quantum states include:
- $|0〉, |1〉$ = computational basis
- $|+〉 = 1/sqrt(2)(|0〉 + |1〉)$ = superposition (plus state)
- $|-〉 = 1/sqrt(2)(|0〉 - |1〉)$ = superposition (minus state)

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

For example, the error $E = X_1 Z_3$ on 3 qubits (X on qubit 1, Z on qubit 3) has binary representation:
$ bold(e)_Q = (bold(x), bold(z)) = ((1,0,0), (0,0,1)) $

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

Quantum codes use double-bracket notation $[[n, k, d]]$:
- $n$ = number of physical qubits
- $k$ = number of logical qubits encoded
- $d$ = code distance (minimum weight of undetectable errors)

Compare to classical $[n, k, d]$ notation (single brackets).

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

A well-known example is the *Toric Code*, which is the hypergraph product of the ring code (cyclic repetition code). From a classical $[n, 1, n]$ ring code, we obtain a quantum $[[2n^2, 2, n]]$ Toric code. Its properties include:
- $(4, 4)$-QLDPC: each stabilizer involves at most 4 qubits
- High threshold (~10.3% with optimal decoder)
- Rate $R = 2/(2n^2) arrow.r 0$ as $n arrow.r infinity$

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

#pagebreak()

= References

#bibliography("references.bib", style: "ieee")

#v(2em)
#align(center)[#text(style: "italic")[End of Lecture Note]]

