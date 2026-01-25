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
  OSD then resolves split beliefs and *forcing a unique solution*. 
  It first choose the basis selection $[S]$ through
  BP soft decisions guide. and then calculate the matrix inversion on $H_([S])$ eliminates ambiguity.

== OSD-0: The Basic Algorithm

Ordered Statistics Decoding (OSD) was introduced by Fossorier and Lin as a soft-decision decoding algorithm for linear block codes that approaches maximum-likelihood performance @fossorier1995soft. The algorithm was later extended with computationally efficient variants @fossorier1996efficient. For quantum LDPC codes, the BP+OSD combination was shown to be remarkably effective by Panteleev and Kalachev, and further developed by Roffe et al. @roffe2020decoding.

#definition[
  *OSD-0* finds a solution by:
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

The GPU-accelerated implementation in `batch_osd.py` realizes OSD-0 through Gaussian elimination rather than explicit matrix inversion. Below is the mapping between the theoretical definition and the code logic.

=== Step 1: Choosing a "Good" Basis (Sorting)

The definition requires selecting basis columns $[S]$ based on BP soft decisions. For quantum error correction, the implementation sorts by *probability descending* rather than reliability $|p - 0.5|$.

```python
# Sort by probability descending (highest probability first)
# High-probability errors become pivots; low-probability errors become free variables
sorted_indices = np.argsort(probs)[::-1]
```

#keypoint[
  *Why probability-based sorting for quantum codes?*

  Traditional OSD uses reliability $|p - 0.5|$, but this fails for quantum codes where:
  - Most qubits have $p approx 0$ (no error) $arrow.r$ reliability $approx 0.5$
  - Identified errors have $p approx 1$ $arrow.r$ reliability $approx 0.5$

  Both have similar reliability despite being very different! Sorting by probability directly places likely errors in $[S]$ (pivots) and unlikely errors in $[T]$ (free variables set to 0).
]

=== Step 2: Solving for Basis Bits (RREF)

Instead of computing $H_([S])^(-1)$ explicitly, the code computes the *Reduced Row Echelon Form* (RREF) of the augmented matrix $[H_"sorted" | bold(s)]$. This is numerically stable and solves the linear system in one pass. For example, consider a parity check matrix $H$ with 3 checks and 6 variables, and syndrome $bold(s) = (1, 0, 1)^T$. Assume BP has already sorted columns by probability (column 0 = highest probability error). Then the initial augmented matrix $[H_"sorted" | bold(s)]$ writes:

$ mat(
  1, 0, 1, 1, 0, 1, |, 1;
  1, 1, 0, 0, 1, 1, |, 0;
  0, 1, 1, 0, 1, 0, |, 1;
) $

*Iteration 1: Process column 0*
- Find pivot: Row 0 has a 1 in column 0 $checkmark$
- Eliminate: Row 1 also has 1 in column 0, so XOR row 1 with row 0

$ mat(
  bold(1), 0, 1, 1, 0, 1, |, 1;
  0, 1, 1, 1, 1, 0, |, 1;
  0, 1, 1, 0, 1, 0, |, 1;
) quad "pivot_cols" = [0] $

*Iteration 2: Process column 1*
- Find pivot: Row 1 has a 1 in column 1 $checkmark$
- Eliminate: Row 2 also has 1 in column 1, so XOR row 2 with row 1

$ mat(
  bold(1), 0, 1, 1, 0, 1, |, 1;
  0, bold(1), 1, 1, 1, 0, |, 1;
  0, 0, 0, 1, 0, 0, |, 0;
) quad "pivot_cols" = [0, 1] $

*Iteration 3: Process column 2*
- Find pivot: No rows have a 1 in column 2 below the current pivot row $times$
- Skip this column (it becomes a free variable)

*Iteration 4: Process column 3*
- Find pivot: Row 2 has a 1 in column 3 $checkmark$
- Eliminate: Rows 0 and 1 have 1s in column 3, XOR both with row 2

$ mat(
  bold(1), 0, 1, 0, 0, 1, |, 1;
  0, bold(1), 1, 0, 1, 0, |, 1;
  0, 0, 0, bold(1), 0, 0, |, 0;
) quad "pivot_cols" = [0, 1, 3] $

*Final Result:* Pivot columns $[S] = {0, 1, 3}$ (basis variables) and free columns $[T] = {2, 4, 5}$ (remainder variables).

In the code, the H matrix is stored as a 2D 'int8' array to fully utilized the GPU's integer arithmetic capabilities.
The fuctiion 'compute_rref' implements Gaussian elimination over $"GF"(2)$:

```python
def _get_rref_cached(self, sorted_indices: np.ndarray, syndrome: np.ndarray):
    # Reorder columns by sorted indices
    H_sorted = self.H[:, sorted_indices]
    # Build augmented matrix [H_sorted | s]
    augmented = np.hstack([H_sorted, syndrome.reshape(-1, 1)]).astype(np.int8)
    # Compute RREF in-place
    pivot_cols = self._compute_rref(augmented)
    return augmented, pivot_cols

def _compute_rref(self, M: np.ndarray) -> List[int]:
    m, n = M.shape
    pivot_row = 0
    pivot_cols = []

    for col in range(n - 1):  # Don't pivot on syndrome column
        if pivot_row >= m:
            break
        # Find a row with 1 in this column
        candidates = np.where(M[pivot_row:, col] == 1)[0]
        if len(candidates) == 0:
            continue  # No pivot in this column

        # Swap to bring pivot to current row
        swap_r = candidates[0] + pivot_row
        if swap_r != pivot_row:
            M[[pivot_row, swap_r]] = M[[swap_r, pivot_row]]

        pivot_cols.append(col)

        # Eliminate all other 1s in this column (XOR in GF(2))
        rows_to_xor = np.where(M[:, col] == 1)[0]
        rows_to_xor = rows_to_xor[rows_to_xor != pivot_row]
        if len(rows_to_xor) > 0:
            M[rows_to_xor, :] ^= M[pivot_row, :]

        pivot_row += 1

    return pivot_cols
```

- `pivot_cols`: The column indices where pivots were found. These form the basis set $[S]$.
- After RREF, the basis submatrix has an identity-like structure, and the syndrome column contains the solution values.

// #keypoint[
//   The RREF algorithm transforms both $H$ and $bold(s)$ simultaneously using the same row operations. After RREF, each row $r$ with pivot column $c$ gives us directly: $e_c = "augmented"[r, -1]$ (the transformed syndrome value in that row).
// ]

=== Step 3: Reading the OSD-0 solution:
Continue from the previous example, the result can be read from the pivot collumns and syndrome column $bold(s)$, as illustated by the following steps:
- Set free variables to zero: $e_2 = e_4 = e_5 = 0$
- Read pivot values from the transformed syndrome column:
  - Row 0: pivot at column 0, syndrome value = 1 $arrow.r e_0 = 1$
  - Row 1: pivot at column 1, syndrome value = 1 $arrow.r e_1 = 1$
  - Row 2: pivot at column 3, syndrome value = 0 $arrow.r e_3 = 0$
- Solution in sorted order then must be $bold(e)_"sorted" = (1, 1, 0, 0, 0, 0)$, and can be verified through:

$ H_"sorted" dot bold(e)_"sorted" = mat(
  1, 0, 1, 1, 0, 1;
  1, 1, 0, 0, 1, 1;
  0, 1, 1, 0, 1, 0;
) dot mat(1; 1; bold(0); 0 ; bold(0); bold(0)) = mat(
  1 plus.o 0;
  1 plus.o 1;
  0 plus.o 1
) = mat(1; 0; 1) = bold(s) quad checkmark $


In the code, the solution is extracted by initializing the solution vector to zeros and updating *only* the pivot positions using the transformed syndrome.


```python
# OSD-0 Solution: Initialize all bits to 0 (remainder bits stay 0)
solution_base = np.zeros(self.num_errors, dtype=np.int8)

# Build pivot-to-row mapping and extract solution
pivot_row_map = {}
for r in range(augmented.shape[0]):
    row_pivots = np.where(augmented[r, :self.num_errors] == 1)[0]
    if len(row_pivots) > 0:
        col = row_pivots[0]
        if col in pivot_cols:
            pivot_row_map[col] = r
            # Pivot bit = transformed syndrome value
            solution_base[col] = augmented[r, -1]
```

- `solution_base = np.zeros(...)`: Ensures $bold(e)_([T]) = bold(0)$ (OSD-0 constraint).
- `augmented[r, -1]`: The transformed syndrome value. Since the basis submatrix is now identity-like, this directly gives $bold(e)_([S])$.

=== Step 4: Inverse Mapping (Unsort)

Finally, the solution is mapped back to the original column ordering:

```python
# Remap from sorted order back to original order
estimated_errors = np.zeros(self.num_errors, dtype=int)
estimated_errors[sorted_indices] = final_solution_sorted
return estimated_errors
```

#figure(
table(
columns: 2,
align: (left, left),
stroke: 0.5pt,
[*Theoretical Step*], [*Code Realization (`batch_osd.py`)*],
[1. Sort by soft decisions], [`np.argsort(probs)[::-1]` (probability descending)],
[2. Select basis $[S]$], [`pivot_cols` from `_compute_rref`],
[3. Matrix inversion], [RREF transforms basis to identity structure],
[4. Solve $bold(e)_([S])$], [`solution_base[col] = augmented[r, -1]`],
[5. Set $bold(e)_([T]) = bold(0)$], [`np.zeros(...)` initialization],
[6. Unsort], [`estimated_errors[sorted_indices] = solution`],
),
caption: [Mapping OSD-0 theory to `batch_osd.py` implementation]
)


== Higher-Order OSD (OSD-$lambda$)

OSD-0 assumes the remainder error bits are zero ($bold(e)_([T]) = bold(0)$). While this provides a valid solution, it forces all "correction" work onto the basis bits $[S]$, which may result in a high-weight (improbable) error pattern.

#definition[
  *Higher-order OSD* improves this by testing non-zero configurations for the remainder bits $bold(e)_([T])$.
  For any chosen hypothesis $bold(e)_([T])$, the corresponding basis bits $bold(e)_([S])$ are uniquely determined to satisfy the syndrome:
  $ bold(e)_([S]) = H_([S])^(-1) dot (bold(s) + H_([T]) dot bold(e)_([T])) $
]

It is straightforward to show that the constructed error $bold(e) = (bold(e)_([S]), bold(e)_([T]))$ always satisfies the parity check equation $H dot bold(e) = bold(s)$ and the OSD-0 is a special case of OSD-$lambda$ when $lambda = 0$.

$ H dot bold(e) &= mat(H_([S]), H_([T])) dot mat(bold(e)_([S]); bold(e)_([T])) = H_([S]) dot bold(e)_([S]) + H_([T]) dot bold(e)_([T]) \
  &= H_([S]) dot [H_([S])^(-1) dot (bold(s) + H_([T]) dot bold(e)_([T]))] + H_([T]) dot bold(e)_([T]) \
  &= I dot (bold(s) + H_([T]) dot bold(e)_([T])) + H_([T]) dot bold(e)_([T]) \
  &= bold(s) + H_([T]) dot bold(e)_([T]) + H_([T]) dot bold(e)_([T]) = bold(s) + bold(0) = bold(s) $

Then the problem change to find the minimum soft-weight solution for the remainder bits $bold(e)_([T])$.
A naive way to do this is to implement an exhaustive search (OSD-E) testing on all $2^(k')$ patterns. This guarantees finding the minimum weight solution.
Unfortunately, the remainder set $[T]$ has size $k' = n - r$, which is exponentially large in the code parameters $n - r$.
 To make this feasible, we restrict the search to the search depth $lambda$, i.e., the *most suspicious* $lambda$ bits in $[T]$ (those with highest error probability among free variables) and accelerate this serch process by using the GPU.
The `batch_osd.py` implementation accelerates OSD-E by evaluating all $2^lambda$ candidates in parallel on GPU. Here is how it works:

*Step 1: Identify Search Columns*

Select the $lambda$ free variables with highest error probability (most suspicious):

```python
# Get free columns (not pivots)
all_cols = set(range(self.num_errors))
free_cols = sorted(list(all_cols - set(pivot_cols)))

# Sort free columns by probability (highest first = most suspicious)
free_cols_with_prob = [(col, probs[sorted_indices[col]]) for col in free_cols]
free_cols_with_prob.sort(key=lambda x: -x[1])

# Select top osd_order free variables for search
search_cols = [col for col, _ in free_cols_with_prob[:osd_order]]
```

*Step 2: Generate All $2^lambda$ Candidates*

```python
# Exhaustive: Generate all 2^k combinations using bit manipulation
num_candidates = 1 << len(search_cols)  # 2^k
candidates_np = np.array([
    [(i >> j) & 1 for j in range(len(search_cols))]
    for i in range(num_candidates)
], dtype=np.int8)
```

*Step 3: Parallel Evaluation on GPU*

All candidates are evaluated simultaneously using batched matrix operations:

```python
def _evaluate_candidates_gpu(self, candidates, augmented, search_cols, probs_sorted, pivot_cols):
    num_candidates = candidates.shape[0]

    # Transfer to GPU
    M_subset = torch.from_numpy(augmented[:, search_cols]).float().to(self.device)
    syndrome_col = torch.from_numpy(augmented[:, -1]).float().to(self.device)

    # Compute modified syndromes for ALL candidates in parallel
    # target_syndrome = (s + M @ e_T) % 2
    target_syndromes = (syndrome_col.unsqueeze(0) + candidates.float() @ M_subset.T) % 2

    # Initialize solution matrix (num_candidates × n)
    cand_solutions = torch.zeros(num_candidates, self.num_errors, device=self.device)

    # Set free variable values from candidates
    cand_solutions[:, search_cols] = candidates.float()

    # Solve for pivot variables using the RREF structure
    for r in range(augmented.shape[0]):
        row_pivots = torch.where(augmented_torch[r, :] == 1)[0]
        if len(row_pivots) > 0:
            pivot_c = row_pivots[0].item()
            if pivot_c in pivot_cols:
                # Pivot value = modified syndrome for this row
                cand_solutions[:, pivot_c] = target_syndromes[:, r]

    # Compute soft-weighted costs and return best solution
    costs = self._compute_soft_weight_gpu(cand_solutions, probs_sorted)
    best_idx = torch.argmin(costs)
    return cand_solutions[best_idx]
```

*Step 4: Soft-Weight Cost Function*

In our code, the OSD uses the *soft-weighted cost* based on log-probabilities @roffe2020decoding:
#definition[
  *Soft-Weighted Cost (Log-Probability Weight).* For an error pattern $bold(e) = (e_1, ..., e_n)$ with bit-wise error probabilities $p_i = P(e_i = 1)$, the soft-weighted cost is:

  $ W_"soft"(bold(e)) = sum_(i : e_i = 1) (-log p_i) = - sum_(i=1)^n e_i dot log p_i $

  Lower cost indicates a more probable error pattern.
]
There actually are several other cost functions that can be used for OSD, such as the Hamming weight, Euclidean distance, and LLR-based weight, as listed in the following table:
#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: 0.5pt,
    [*Cost Function*], [*Formula*], [*Properties*],
    [Hamming Weight @hamming1950error], [$W_H(bold(e)) = sum_i e_i$], [Counts flipped bits; ignores probabilities],
    [Soft Weight (Log-Prob) @roffe2020decoding], [$W_"soft"(bold(e)) = -sum_i e_i log p_i$], [Weights by $-log p_i$; approximates ML],
    [Euclidean Distance @forney1966generalized], [$d_E^2 = sum_i (r_i - c_i)^2$], [For AWGN channels with continuous signals],
    [LLR-Based Weight @hagenauer1996iterative], [$W_"LLR"(bold(e)) = sum_i e_i |L_i|$], [Uses log-likelihood ratios $L_i = log(p_i / (1-p_i))$],
  ),
  caption: [Comparison of cost functions for selecting the best error pattern]
)

Then why soft weight is preferred for BP+OSD? 
This is because the soft weight approximates the Maximum-Likelihood Decoding (ML) objective @yue2020revisit. The ML decoder selects the error pattern $bold(e)^*$ that maximizes the posterior probability 
$ bold(e)^* = arg max_bold(e) P(bold(e) | "syndrome"). $
 Taking the logarithm, which is a monotonic transformation, and suppose the error $e_i$ are independent, each with probability $P(e_i = 1) = p_i$, this becomes:
   $ bold(e)^* = arg max_bold(e) sum_i [e_i log p_i + (1-e_i) log(1-p_i)] $
   For sparse errors where most $e_i = 0$, minimizing $W_"soft"(bold(e)) = -sum_i e_i log p_i$ closely approximates the ML objective.
#keypoint[
  *Question:* Why not use the Hamming weight or Euclidean distance as the cost function?
]
In the code, the soft-weight cost function is implemented as follows:
```python
def _compute_soft_weight_gpu(self, solutions, probs):
    # Clip to avoid log(0)
    probs_clipped = torch.clamp(probs, 1e-10, 1 - 1e-10)
    # Log-probability weights: -log(p) penalizes flipping low-probability bits
    log_weights = -torch.log(probs_clipped)
    # Total cost = sum of weights for flipped bits
    costs = (solutions * log_weights).sum(dim=1)
    return costs
```
== Combination Sweep Strategy (OSD-CS)

To allow for a larger search depth (e.g., $lambda approx 50-100$) without exponential cost, we use the *combination sweep* strategy, first proposed for reducing error floors in classical LDPC codes and adapted for quantum codes by Roffe et al. @roffe2020decoding.

#definition[
  *OSD-CS* assumes the true error pattern on the remainder bits is *sparse*. Instead of checking *all* $2^lambda$ patterns on $lambda$ most suspicious bits, it only checks those with low Hamming weight (w = 0,1,2). Exausted on those strings only take $C_lambda^0 + C_lambda^1 + C_lambda^2 = 1 + lambda + lambda(lambda-1)/2$ candidates.
With $lambda = 60$: approximately $1 + k + 1770$ configurations (vs $2^k$ for exhaustive search!)

  *Algorithm Steps:*
  1. *Sort:* Select the $lambda$ most suspicious positions in $[T]$ (highest probability among free variables).
  2. *Sweep:* Generate candidate vectors $bold(e)_([T])$ with:
     - *Weight 0:* The zero vector (equivalent to OSD-0).
     - *Weight 1:* All single-bit flips among the $lambda$ bits.
     - *Weight 2:* All pairs of bit flips among the $lambda$ bits.
  3. *Select:* Calculate $bold(e)_([S])$ for each candidate, compute the soft-weighted cost, and pick the best one.
]

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    stroke: 0.5pt,
    [*Method*], [*Complexity*], [*Use Case*],
    [OSD-0], [$O(1)$], [Fastest, baseline performance],
    [OSD-E (Exhaustive)], [$O(2^lambda)$], [Optimal for small $lambda$ ($lt.eq 15$)],
    [OSD-CS (Comb. Sweep)], [$O(lambda^2)$], [Near-optimal for large $lambda$ ($approx 60$)],
  ),
  caption: [Comparison of OSD Search Strategies]
)
  *Why OSD-CS works for Quantum Codes?*
  In the low-error regime relevant for QEC, it is statistically very unlikely that the optimal solution requires flipping 3+ bits in the specific subset of "uncertain" remainder bits.
  Checking only weights 0, 1, and 2 captures the vast majority of likely error configurations while reducing complexity from exponential to polynomial (quadratic) @roffe2020decoding.

#definition[
  *Combination sweep* is a greedy search testing configurations by likelihood:

  #enum(
    [*Sort remainder bits:* Order bits in $[T]$ by error probability (most likely first)],
    [*Test weight-0:* The zero vector (OSD-0 baseline)],
    [*Test weight-1:* Set each single bit in $bold(e)_([T])$ to 1 (all $k$ possibilities)],
    [*Test weight-2:* Set each pair among the first $lambda$ bits to 1]
  )

  Keep the minimum soft-weight solution found.
]



=== OSD-CS Implementation in `batch_osd.py`

The GPU-accelerated implementation realizes OSD-CS by explicitly generating sparse error patterns (weights 0, 1, and 2) instead of iterating through all binary combinations.

*Step 1: Generating Sparse Candidates*

The `_generate_osd_cs_candidates` method generates candidate vectors $bold(e)_([T])$ with structured loops:

```python
def _generate_osd_cs_candidates(self, k: int, osd_order: int) -> np.ndarray:
    """Generate OSD-CS (Combination Sweep) candidate strings."""
    candidates = []

    # Weight 0: Zero vector (OSD-0 baseline)
    candidates.append(np.zeros(k, dtype=np.int8))

    # Weight 1: Single-bit flips (k candidates)
    for i in range(k):
        candidate = np.zeros(k, dtype=np.int8)
        candidate[i] = 1
        candidates.append(candidate)

    # Weight 2: Two-bit flips (limited to osd_order)
    for i in range(min(osd_order, k)):
        for j in range(i + 1, min(osd_order, k)):
            candidate = np.zeros(k, dtype=np.int8)
            candidate[i] = 1
            candidate[j] = 1
            candidates.append(candidate)

    return np.array(candidates, dtype=np.int8)
```

- `np.zeros(k)`: The "baseline" hypothesis (OSD-0 solution).
- `range(k)` loop: Adds $k$ candidates, each with a single bit flip at index `i`.
- Nested `range(limit)` loop: Adds $binom(lambda, 2)$ candidates, representing pairs of flips at indices $(i, j)$.


*Step 2: Integration into the Solve Loop*

The `solve` method switches between OSD-E and OSD-CS based on the `osd_method` parameter:

```python
# Generate candidates based on method
if osd_method == 'combination_sweep':
    # OSD-CS: O(λ²) sparse candidates
    candidates_np = self._generate_osd_cs_candidates(len(search_cols), osd_order)
else:
    # Exhaustive: O(2^k) all combinations
    num_candidates = 1 << len(search_cols)
    candidates_np = np.array([[(i >> j) & 1 for j in range(len(search_cols))]
                             for i in range(num_candidates)], dtype=np.int8)

# Transfer to GPU and evaluate all candidates in parallel
candidates = torch.from_numpy(candidates_np).to(self.device)
best_solution_sorted = self._evaluate_candidates_gpu(
    candidates, augmented, search_cols, probs_sorted, pivot_cols
)
```

Both methods use the same GPU evaluation function---only the candidate generation differs.

*Complexity Comparison*

#figure(
table(
columns: 3,
align: (left, center, center),
stroke: 0.5pt,
[*Method*], [*Candidates*], [*Example ($lambda = 15$)*],
[OSD-E], [$2^lambda$], [$32768$],
[OSD-CS], [$1 + k + binom(lambda, 2)$], [$approx 1 + k + 105$],
),
caption: [Number of candidates for OSD-E vs OSD-CS]
)

#figure(
table(
columns: 2,
align: (left, left),
stroke: 0.5pt,
[*Weight Class*], [*Code Realization*],
[Weight 0], [`candidates.append(np.zeros(k))`],
[Weight 1], [`for i in range(k): candidate[i] = 1`],
[Weight 2], [`for i in range(limit): for j in range(i+1, limit): ...`],
),
caption: [Mapping OSD-CS theory to `batch_osd.py` implementation]
)

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

== Surface Codes: A Comprehensive Introduction

The *surface code* @kitaev2003fault @bravyi1998quantum is the most studied and practically promising quantum error-correcting code. It belongs to the family of *topological codes*, where quantum information is encoded in global, topological degrees of freedom that are inherently protected from local errors.

=== Stabilizer Formalism for Surface Codes

#definition[
  A *stabilizer code* is defined by an Abelian subgroup $cal(S)$ of the $n$-qubit Pauli group $"Pauli"(n)$, generated by a set of independent generators $bb(g) = {g_1, dots, g_(n-k)}$.
  
  The *code space* $V(cal(S))$ is the simultaneous $+1$-eigenspace of all stabilizers:
  $ V(cal(S)) = {|psi angle.r : g_i |psi angle.r = |psi angle.r, forall g_i in bb(g)} $
  
  This subspace has dimension $2^k$, encoding $k$ logical qubits into $n$ physical qubits.
]

For a Pauli error $E$ acting on a code state $|c angle.r in V(cal(S))$:
$ g_i E |c angle.r = (-1)^(l_i(E)) E |c angle.r $

where the *syndrome bit* $l_i(E)$ indicates whether $E$ anticommutes with stabilizer $g_i$. Measuring all stabilizers yields the syndrome vector $bold(l)(E) = (l_1, dots, l_(n-k))$, which identifies the error's equivalence class.

#keypoint[
  *Error correction criterion:* A stabilizer code corrects a set of errors $bb(E)$ if and only if:
  $ forall E in bb(E): quad bb(E) sect (E dot bold(C)(cal(S)) - E dot cal(S)) = emptyset $
  
  where $bold(C)(cal(S))$ is the *centralizer* of $cal(S)$ (Pauli operators commuting with all stabilizers). Errors in $E dot cal(S)$ are *degenerate* (equivalent to $E$), while errors in $bold(C)(cal(S)) - cal(S)$ are *logical operators* that transform between codewords.
]

=== The Toric Code: Surface Code on a Torus

#definition[
  The *toric code* @kitaev2003fault is defined on a square lattice embedded on a torus, with qubits placed on edges. For an $L times L$ lattice, the code has parameters $[[2L^2, 2, L]]$.
  
  Stabilizer generators are of two types:
  - *Star operators* $A_v = product_(e in "star"(v)) X_e$: product of $X$ on all edges meeting at vertex $v$
  - *Plaquette operators* $B_p = product_(e in "boundary"(p)) Z_e$: product of $Z$ on all edges bounding plaquette $p$
]

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // Draw a 3x3 grid representing the toric code lattice
    let grid_size = 3
    let spacing = 1.5
    
    // Vertices (circles)
    for i in range(grid_size) {
      for j in range(grid_size) {
        circle((i * spacing, j * spacing), radius: 0.08, fill: black)
      }
    }
    
    // Horizontal edges with qubits
    for i in range(grid_size - 1) {
      for j in range(grid_size) {
        line((i * spacing + 0.1, j * spacing), ((i + 1) * spacing - 0.1, j * spacing), stroke: gray)
        circle(((i + 0.5) * spacing, j * spacing), radius: 0.12, fill: blue.lighten(60%))
      }
    }
    
    // Vertical edges with qubits
    for i in range(grid_size) {
      for j in range(grid_size - 1) {
        line((i * spacing, j * spacing + 0.1), (i * spacing, (j + 1) * spacing - 0.1), stroke: gray)
        circle((i * spacing, (j + 0.5) * spacing), radius: 0.12, fill: blue.lighten(60%))
      }
    }
    
    // Highlight a star operator (red X)
    let star_x = 1.5
    let star_y = 1.5
    circle((star_x, star_y), radius: 0.25, stroke: red + 2pt, fill: red.lighten(90%))
    content((star_x, star_y), text(size: 8pt, fill: red)[$A_v$])
    
    // Highlight a plaquette operator (blue Z)
    rect((0.5 * spacing + 0.15, 0.5 * spacing + 0.15), (1.5 * spacing - 0.15, 1.5 * spacing - 0.15), 
         stroke: blue + 2pt, fill: blue.lighten(90%))
    content((spacing, spacing), text(size: 8pt, fill: blue)[$B_p$])
    
    // Legend
    content((4.5, 2.5), text(size: 8pt)[Qubits on edges])
    content((4.5, 2), text(size: 8pt, fill: red)[$A_v = X X X X$ (star)])
    content((4.5, 1.5), text(size: 8pt, fill: blue)[$B_p = Z Z Z Z$ (plaquette)])
  }),
  caption: [Toric code lattice: qubits reside on edges, star operators ($A_v$) act on edges meeting at vertices, plaquette operators ($B_p$) act on edges surrounding faces.]
)

The star and plaquette operators satisfy:
- *Commutativity:* $[A_v, A_(v')] = [B_p, B_(p')] = [A_v, B_p] = 0$ for all $v, v', p, p'$
- *Redundancy:* $product_v A_v = product_p B_p = bb(1)$ (only $2L^2 - 2$ independent generators)
- *Topological protection:* Logical operators correspond to non-contractible loops on the torus

#keypoint[
  *Topological interpretation:* Errors create *anyonic excitations* at their endpoints:
  - $X$ errors create pairs of $m$-anyons (plaquette violations)
  - $Z$ errors create pairs of $e$-anyons (star violations)
  
  A logical error occurs when an anyon pair is created, transported around a non-contractible cycle, and annihilated—this cannot be detected by local stabilizer measurements.
]

=== Planar Surface Code with Boundaries

For practical implementations, we use a *planar* version with open boundaries @bravyi1998quantum @dennis2002topological:

#definition[
  The *planar surface code* is defined on a square patch with two types of boundaries:
  - *Rough boundaries* (top/bottom): where $X$-type stabilizers are truncated
  - *Smooth boundaries* (left/right): where $Z$-type stabilizers are truncated
  
  A distance-$d$ planar code has parameters $[[d^2 + (d-1)^2, 1, d]]$ for storing one logical qubit, or approximately $[[2d^2, 1, d]]$.
]

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // Draw planar surface code with boundaries
    let size = 4
    let s = 0.9
    
    // Background to show boundary types
    rect((-0.3, -0.3), (size * s + 0.3, size * s + 0.3), fill: white, stroke: none)
    
    // Rough boundaries (top and bottom) - red
    line((-0.2, -0.2), (size * s + 0.2, -0.2), stroke: red + 3pt)
    line((-0.2, size * s + 0.2), (size * s + 0.2, size * s + 0.2), stroke: red + 3pt)
    
    // Smooth boundaries (left and right) - blue  
    line((-0.2, -0.2), (-0.2, size * s + 0.2), stroke: blue + 3pt)
    line((size * s + 0.2, -0.2), (size * s + 0.2, size * s + 0.2), stroke: blue + 3pt)
    
    // Draw checkerboard pattern for X and Z stabilizers
    for i in range(size) {
      for j in range(size) {
        let x = i * s
        let y = j * s
        let color = if calc.rem(i + j, 2) == 0 { rgb("#ffe0e0") } else { rgb("#e0e0ff") }
        rect((x, y), (x + s, y + s), fill: color, stroke: gray + 0.5pt)
      }
    }
    
    // Qubits at vertices
    for i in range(size + 1) {
      for j in range(size + 1) {
        circle((i * s, j * s), radius: 0.08, fill: black)
      }
    }
    
    // Logical operators
    // Logical X - horizontal path (rough to rough)
    for i in range(size) {
      circle((i * s + s/2, 0), radius: 0.12, fill: green.lighten(40%), stroke: green + 1.5pt)
    }
    
    // Logical Z - vertical path (smooth to smooth)
    for j in range(size) {
      circle((0, j * s + s/2), radius: 0.12, fill: purple.lighten(40%), stroke: purple + 1.5pt)
    }
    
    // Legend
    content((5.5, 3), text(size: 8pt, fill: red)[Rough boundary])
    content((5.5, 2.5), text(size: 8pt, fill: blue)[Smooth boundary])
    content((5.5, 2), text(size: 8pt, fill: green)[$X_L$: rough $arrow.r$ rough])
    content((5.5, 1.5), text(size: 8pt, fill: purple)[$Z_L$: smooth $arrow.r$ smooth])
  }),
  caption: [Planar surface code with rough (red) and smooth (blue) boundaries. Logical $X_L$ connects rough boundaries; logical $Z_L$ connects smooth boundaries.]
)

The boundary conditions determine the logical operators:
- *Logical $X_L$:* String of $X$ operators connecting the two rough boundaries
- *Logical $Z_L$:* String of $Z$ operators connecting the two smooth boundaries

#keypoint[
  *Code distance:* The minimum weight of a logical operator equals $d$, the lattice width. To cause a logical error, noise must create a string of errors spanning the entire code—an event exponentially suppressed for $p < p_"th"$.
]

=== The Rotated Surface Code

The *rotated surface code* @bombin2007optimal @tomita2014low is a more hardware-efficient variant:

#definition[
  The *rotated surface code* is obtained by rotating the standard surface code lattice by 45°. Qubits are placed on vertices of a checkerboard pattern, with:
  - $X$-type stabilizers on one color (e.g., white squares)
  - $Z$-type stabilizers on the other color (e.g., gray squares)
  
  For distance $d$, the code has parameters $[[d^2, 1, d]]$—roughly half the qubits of the standard planar code at the same distance.
]

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    let d = 3  // distance
    let s = 1.0  // spacing
    
    // Draw the rotated lattice (checkerboard)
    for i in range(d) {
      for j in range(d) {
        let x = i * s
        let y = j * s
        // Alternate X and Z stabilizers
        if calc.rem(i + j, 2) == 0 {
          rect((x - s/2, y - s/2), (x + s/2, y + s/2), fill: rgb("#ffe0e0"), stroke: gray + 0.3pt)
          content((x, y - 0.05), text(size: 7pt, fill: red.darken(30%))[$X$])
        } else {
          rect((x - s/2, y - s/2), (x + s/2, y + s/2), fill: rgb("#e0e0ff"), stroke: gray + 0.3pt)
          content((x, y - 0.05), text(size: 7pt, fill: blue.darken(30%))[$Z$])
        }
      }
    }
    
    // Data qubits at corners of squares
    for i in range(d + 1) {
      for j in range(d + 1) {
        // Only place qubits that touch at least one stabilizer
        if (i > 0 or j > 0) and (i < d or j < d) and (i > 0 or j < d) and (i < d or j > 0) {
          let x = (i - 0.5) * s
          let y = (j - 0.5) * s
          circle((x, y), radius: 0.12, fill: blue.lighten(50%), stroke: black + 0.8pt)
        }
      }
    }
    
    // Labels
    content((3.5, 2), text(size: 9pt)[Distance $d = 3$])
    content((3.5, 1.5), text(size: 9pt)[$[[9, 1, 3]]$ code])
    content((3.5, 1), text(size: 9pt)[9 data qubits])
    content((3.5, 0.5), text(size: 9pt)[8 ancilla qubits])
  }),
  caption: [Rotated surface code with $d = 3$. Data qubits (blue circles) sit at corners where stabilizer plaquettes meet. This is also known as the Surface-17 code (9 data + 8 ancilla qubits).]
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    stroke: 0.5pt,
    inset: 8pt,
    [*Distance $d$*], [*Data qubits*], [*Ancilla qubits*], [*Total qubits*], [*Code*],
    [3], [$9$], [$8$], [$17$], [Surface-17],
    [5], [$25$], [$24$], [$49$], [Surface-49],
    [7], [$49$], [$48$], [$97$], [Surface-97],
    [$d$], [$d^2$], [$d^2 - 1$], [$2d^2 - 1$], [General],
  ),
  caption: [Rotated surface code parameters for various distances]
)

=== Syndrome Extraction Circuits

Practical surface code implementations require *syndrome extraction circuits* that measure stabilizers without destroying the encoded information @fowler2012surface:

#definition[
  *Syndrome extraction* uses ancilla qubits to measure stabilizers via the Hadamard test:
  1. Initialize ancilla in $|0 angle.r$ (for $Z$-stabilizers) or $|+ angle.r$ (for $X$-stabilizers)
  2. Apply controlled operations between ancilla and data qubits
  3. Measure ancilla to obtain syndrome bit
  
  The measurement outcome indicates whether the stabilizer eigenvalue is $+1$ (result 0) or $-1$ (result 1).
]

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // Draw syndrome extraction circuit schematic
    content((0, 3), text(size: 9pt)[*$X$-stabilizer measurement:*])
    
    // Ancilla line
    line((1, 2), (6, 2), stroke: gray)
    content((0.5, 2), text(size: 8pt)[$|0 angle.r$])
    rect((1.2, 1.8), (1.8, 2.2), fill: yellow.lighten(70%))
    content((1.5, 2), text(size: 8pt)[$H$])
    
    // Data qubit lines
    for i in range(4) {
      let y = 0.8 - i * 0.4
      line((1, y), (6, y), stroke: gray)
      content((0.5, y), text(size: 8pt)[$q_#(i+1)$])
    }
    
    // CNOT gates (ancilla controls)
    for (i, x) in ((0, 2.5), (1, 3.2), (2, 3.9), (3, 4.6)) {
      let y = 0.8 - i * 0.4
      circle((x, 2), radius: 0.08, fill: black)
      line((x, 2 - 0.08), (x, y + 0.12))
      circle((x, y), radius: 0.12, stroke: black + 1pt)
      line((x - 0.12, y), (x + 0.12, y))
      line((x, y - 0.12), (x, y + 0.12))
    }
    
    // Final Hadamard and measurement
    rect((5.2, 1.8), (5.8, 2.2), fill: yellow.lighten(70%))
    content((5.5, 2), text(size: 8pt)[$H$])
    rect((6.2, 1.7), (6.8, 2.3), fill: gray.lighten(70%))
    content((6.5, 2), text(size: 8pt)[$M$])
    
    // Z-stabilizer label
    content((0, -1.2), text(size: 9pt)[*$Z$-stabilizer:* Replace $H$ gates with identity, CNOT targets become controls])
  }),
  caption: [Syndrome extraction circuit for a weight-4 $X$-stabilizer. The ancilla mediates the measurement without collapsing the encoded state.]
)

#keypoint[
  *Hook errors and scheduling:* The order of CNOT gates matters! A single fault in the syndrome extraction circuit can propagate to multiple data qubits, creating *hook errors*. Careful scheduling (e.g., the "Z-shape" or "N-shape" order) minimizes error propagation while allowing parallel $X$ and $Z$ syndrome extraction.
]

=== Repeated Syndrome Measurement

A single syndrome measurement can itself be faulty. To achieve fault tolerance, we perform *multiple rounds* of syndrome extraction:

#definition[
  In *repeated syndrome measurement* with $r$ rounds:
  1. Measure all stabilizers $r$ times (typically $r = d$ for distance-$d$ code)
  2. Track syndrome *changes* between consecutive rounds
  3. Decode using the full spacetime syndrome history
  
  Syndrome changes form a 3D structure: 2D spatial syndrome + 1D time axis.
]

This creates a *3D decoding problem*:
- *Space-like errors:* Pauli errors on data qubits appear as pairs of adjacent syndromes in space
- *Time-like errors:* Measurement errors appear as pairs of syndromes in time at the same location
- *Hook errors:* Correlated space-time error patterns from circuit faults

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // Draw 3D spacetime diagram
    let dx = 0.8
    let dy = 0.5
    let dz = 0.7
    
    // Time slices
    for t in range(4) {
      let offset_x = t * 0.3
      let offset_y = t * dz
      
      // Grid for this time slice
      for i in range(3) {
        for j in range(3) {
          let x = i * dx + offset_x
          let y = j * dy + offset_y
          circle((x, y), radius: 0.06, fill: gray.lighten(50%))
        }
      }
      
      // Connect to form grid
      for i in range(2) {
        for j in range(3) {
          let x1 = i * dx + offset_x
          let x2 = (i + 1) * dx + offset_x
          let y = j * dy + offset_y
          line((x1 + 0.06, y), (x2 - 0.06, y), stroke: gray + 0.5pt)
        }
      }
      for i in range(3) {
        for j in range(2) {
          let x = i * dx + offset_x
          let y1 = j * dy + offset_y
          let y2 = (j + 1) * dy + offset_y
          line((x, y1 + 0.06), (x, y2 - 0.06), stroke: gray + 0.5pt)
        }
      }
      
      content((2.8 + offset_x, 0 + offset_y), text(size: 7pt)[$t = #t$])
    }
    
    // Highlight some syndrome events
    circle((0.8 + 0.3, 0.5 + 0.7), radius: 0.1, fill: red)
    circle((0.8 + 0.6, 0.5 + 1.4), radius: 0.1, fill: red)
    line((0.8 + 0.3, 0.5 + 0.7 + 0.1), (0.8 + 0.6, 0.5 + 1.4 - 0.1), stroke: red + 1.5pt)
    
    // Space-like error
    circle((0 + 0.9, 1 + 2.1), radius: 0.1, fill: blue)
    circle((0.8 + 0.9, 1 + 2.1), radius: 0.1, fill: blue)
    line((0 + 0.9 + 0.1, 1 + 2.1), (0.8 + 0.9 - 0.1, 1 + 2.1), stroke: blue + 1.5pt)
    
    // Legend
    content((4.5, 2.5), text(size: 8pt, fill: red)[Time-like error])
    content((4.5, 2.1), text(size: 8pt, fill: red)[(measurement fault)])
    content((4.5, 1.6), text(size: 8pt, fill: blue)[Space-like error])
    content((4.5, 1.2), text(size: 8pt, fill: blue)[(data qubit fault)])
  }),
  caption: [Spacetime syndrome history. Time-like edges (red) represent measurement errors; space-like edges (blue) represent data qubit errors.]
)

=== Decoders for Surface Codes

Several decoding algorithms exist for surface codes, with different trade-offs between accuracy and speed:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, left),
    stroke: 0.5pt,
    inset: 8pt,
    [*Decoder*], [*Complexity*], [*Threshold*], [*Notes*],
    [Maximum Likelihood], [$O(n^2)$ to $hash P$-hard], [$tilde 10.3%$], [Optimal but often intractable],
    [MWPM @dennis2002topological], [$O(n^3)$], [$tilde 10.3%$], [Near-optimal, polynomial time],
    [Union-Find @delfosse2021almost], [$O(n alpha(n))$], [$tilde 9.9%$], [Nearly linear, practical],
    [BP+OSD @roffe2020decoding], [$O(n^2)$], [$tilde 7-8%$], [General QLDPC decoder],
    [Neural Network], [Varies], [$tilde 10%$], [Learning-based, fast inference],
  ),
  caption: [Comparison of surface code decoders. Thresholds shown are for phenomenological noise; circuit-level thresholds are typically $0.5$--$1%$.]
)

The *Minimum Weight Perfect Matching (MWPM)* decoder @dennis2002topological exploits the surface code structure:
1. Construct a complete graph with syndrome defects as vertices
2. Edge weights are negative log-likelihoods of error chains
3. Find minimum-weight perfect matching using Edmonds' blossom algorithm
4. Infer error chain from matched pairs

#keypoint[
  *Why MWPM works for surface codes:* The mapping from errors to syndromes has a special structure—each error creates exactly two syndrome defects at its endpoints. Finding the most likely error pattern reduces to pairing up defects optimally, which is exactly the minimum-weight perfect matching problem.
]

=== Threshold and Scaling Behavior

The surface code exhibits a *threshold* phenomenon:

#definition[
  The *error threshold* $p_"th"$ is the physical error rate below which:
  $ lim_(d arrow.r infinity) p_L(d, p) = 0 quad "for" p < p_"th" $
  
  where $p_L(d, p)$ is the logical error rate for distance $d$ at physical error rate $p$.
  
  Below threshold, the logical error rate scales as:
  $ p_L approx A (p / p_"th")^(ceil(d\/2)) $
  
  for some constant $A$, achieving *exponential suppression* with increasing distance.
]

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: 0.5pt,
    inset: 8pt,
    [*Noise Model*], [*Threshold*], [*Reference*],
    [Code capacity (perfect measurement)], [$tilde 10.3%$], [Dennis et al. 2002],
    [Phenomenological (noisy measurement)], [$tilde 2.9$--$3.3%$], [Wang et al. 2003],
    [Circuit-level depolarizing], [$tilde 0.5$--$1%$], [Various],
    [Circuit-level with leakage], [$tilde 0.3$--$0.5%$], [Various],
  ),
  caption: [Surface code thresholds under different noise models. Circuit-level noise is most realistic but has the lowest threshold.]
)

=== Experimental Realizations

The surface code has been demonstrated in multiple experimental platforms @google2023suppressing @acharya2024quantum:

#keypoint[
  *Google Quantum AI (2023):* Demonstrated that increasing code distance from $d = 3$ to $d = 5$ reduces logical error rate by a factor of $tilde 2$, providing the first evidence of *below-threshold* operation in a surface code.
  
  *Google Quantum AI (2024):* Achieved logical error rates of $0.14%$ per round with $d = 7$ surface code on the Willow processor, demonstrating clear exponential suppression with distance.
]

Key experimental milestones include:
- *2021:* First demonstration of repeated error correction cycles (Google, IBM)
- *2023:* First evidence of exponential error suppression with distance (Google)
- *2024:* Below-threshold operation with high-distance codes (Google Willow)

=== Fault-Tolerant Operations via Lattice Surgery

Universal fault-tolerant quantum computation requires operations beyond error correction. *Lattice surgery* @horsman2012surface @litinski2019game enables logical gates by merging and splitting surface code patches:

#definition[
  *Lattice surgery* performs logical operations by:
  - *Merge:* Join two surface code patches by measuring joint stabilizers along their boundary
  - *Split:* Separate a merged patch by measuring individual stabilizers
  
  These operations implement logical Pauli measurements, enabling Clifford gates and, with magic state distillation, universal computation.
]

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // Two separate patches
    rect((0, 0), (1.5, 1.5), fill: blue.lighten(80%), stroke: blue + 1pt)
    content((0.75, 0.75), text(size: 10pt)[$|psi_1 angle.r$])
    
    rect((2.5, 0), (4, 1.5), fill: green.lighten(80%), stroke: green + 1pt)
    content((3.25, 0.75), text(size: 10pt)[$|psi_2 angle.r$])
    
    // Arrow
    line((4.5, 0.75), (5.5, 0.75), mark: (end: ">"), stroke: 1.5pt)
    content((5, 1.1), text(size: 8pt)[Merge])
    
    // Merged patch
    rect((6, 0), (9, 1.5), fill: purple.lighten(80%), stroke: purple + 1pt)
    content((7.5, 0.75), text(size: 10pt)[$|psi_1 psi_2 angle.r$ entangled])
    
    // Measurement result
    content((7.5, -0.4), text(size: 8pt)[Measures $Z_1 Z_2$ or $X_1 X_2$])
  }),
  caption: [Lattice surgery: merging two surface code patches measures a joint logical Pauli operator, entangling the encoded qubits.]
)

The merge operation effectively measures:
- *Rough merge* (along rough boundaries): Measures $X_1 X_2$
- *Smooth merge* (along smooth boundaries): Measures $Z_1 Z_2$

Combined with single-qubit Cliffords and magic state injection, lattice surgery enables universal fault-tolerant quantum computation entirely within the 2D surface code framework.

#pagebreak()

== Manifest of BP+OSD threshold analysis
In this section, we implement the BP+OSD decoder on the rotated surface code datasets. The end-to-end workflow consists of three stages: (1) generating detector error models from noisy circuits, (2) building the parity check matrix with hyperedge merging, and (3) estimating logical error rates using soft XOR probability chains.

=== Step 1: Generating Rotated Surface Code DEM Files

The first step is to generate a *Detector Error Model (DEM)* from a noisy quantum circuit using *Stim*. The DEM captures the probabilistic relationship between physical errors and syndrome patterns.

#definition[
  A *Detector Error Model (DEM)* is a list of *error mechanisms*, each specifying a probability $p$ of occurrence, a set of *detectors* (syndrome bits) that flip when the error occurs, and optionally, *logical observables* that flip when the error occurs.
]

We use Stim's built-in circuit generator to create rotated surface code memory experiments with circuit-level depolarizing noise:

```python
import stim

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=d,        # Code distance
    rounds=r,          # Number of syndrome measurement rounds
    after_clifford_depolarization=p,      # Noise after gates
    before_round_data_depolarization=p,   # Noise on idle qubits
    before_measure_flip_probability=p,    # Measurement errors
    after_reset_flip_probability=p,       # Reset errors
)

# Extract DEM from circuit
dem = circuit.detector_error_model(decompose_errors=True)
```

The DEM output uses a compact text format. Key elements include:

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    stroke: 0.5pt,
    inset: 8pt,
    [*Syntax*], [*Meaning*],
    [`error(0.01) D0 D1`], [Error with $p=0.01$ that triggers detectors $D_0$ and $D_1$],
    [`error(0.01) D0 D1 ^ D2`], [*Correlated error*: triggers ${D_0, D_1}$ AND ${D_2}$ simultaneously],
    [`error(0.01) D0 L0`], [Error that triggers $D_0$ and flips logical observable $L_0$],
    [`detector D0`], [Declares detector $D_0$ (syndrome bit)],
    [`logical_observable L0`], [Declares logical observable $L_0$],
  ),
  caption: [DEM syntax elements. The `^` separator indicates correlated fault mechanisms.]
)

#keypoint[
  The `^` *separator* is critical for correct decoding. In `error(p) D0 D1 ^ D2`, the fault triggers *both* patterns ${D_0, D_1}$ and ${D_2}$ simultaneously with probability $p$. These must be treated as separate columns in the parity check matrix $H$, each with the same probability $p$.
]

=== Step 2: Building the Parity Check Matrix $H$

Converting the DEM to a parity check matrix $H$ for BP decoding requires two critical processing stages.

==== Stage 1: Separator Splitting

DEM errors with `^` separators represent correlated faults that trigger multiple detector patterns simultaneously. These must be split into *separate columns* in $H$:

#keypoint[
  *Example:* Consider `error(0.01) D0 D1 ^ D2 L0`. This splits into two components:
  - Component 1: detectors $= {D_0, D_1}$, observables $= {}$, probability $= 0.01$
  - Component 2: detectors $= {D_2}$, observables $= {L_0}$, probability $= 0.01$
  
  Each component becomes a *separate column* in the $H$ matrix with the same probability.
]

The splitting algorithm (from `_split_error_by_separator`):

```python
def _split_error_by_separator(targets):
    components = []
    current_detectors, current_observables = [], []
    
    for t in targets:
        if t.is_separator():  # ^ found
            components.append({
                "detectors": current_detectors,
                "observables": current_observables
            })
            current_detectors, current_observables = [], []
        elif t.is_relative_detector_id():
            current_detectors.append(t.val)
        elif t.is_logical_observable_id():
            current_observables.append(t.val)
    
    # Don't forget the last component
    components.append({"detectors": current_detectors, 
                       "observables": current_observables})
    return components
```

==== Stage 2: Hyperedge Merging

After splitting, errors with *identical detector patterns* are merged into single *hyperedges*. This is essential because:
1. Errors with identical syndromes are *indistinguishable* to the decoder
2. Detectors are XOR-based: two errors triggering the same detector cancel out
3. Merging reduces the factor graph size and improves threshold performance

#definition[
  *Hyperedge Merging:* When two error mechanisms have identical detector patterns, their probabilities are combined using the *XOR formula*:
  $ p_"combined" = p_1 + p_2 - 2 p_1 p_2 $
  
  This formula computes $P("odd number of errors fire") = P(A xor B)$.
]

#proof[
  For independent errors $A$ and $B$:
  $ P(A xor B) &= P(A) dot (1 - P(B)) + P(B) dot (1 - P(A)) \
               &= P(A) + P(B) - 2 P(A) P(B) $
  
  This is exactly the probability that an *odd* number of the two errors occurs, which determines the net syndrome flip (since two flips cancel).
]

For observable flip tracking, we compute the *conditional probability* $P("obs flip" | "hyperedge fires")$:

```python
# When merging error with probability prob into existing hyperedge:
if has_obs_flip:
    # New error flips observable: XOR with existing flip probability
    obs_prob_new = obs_prob_old * (1 - prob) + prob * (1 - obs_prob_old)
else:
    # New error doesn't flip observable
    obs_prob_new = obs_prob_old * (1 - prob)

# Store conditional probability: P(obs flip | hyperedge fires)
obs_flip[j] = obs_prob / p_combined
```

#figure(
  table(
    columns: (auto, auto, auto),
    align: (center, center, center),
    stroke: 0.5pt,
    inset: 8pt,
    [*Mode*], [*$H$ Columns (d=3)*], [*Description*],
    [No split, no merge], [$tilde 286$], [Raw DEM errors as columns],
    [Split only], [$tilde 556$], [After `^` separator splitting],
    [Split + merge (optimal)], [$tilde 400$], [After hyperedge merging],
  ),
  caption: [Effect of separator splitting and hyperedge merging on $H$ matrix size for $d=3$ rotated surface code. The split+merge approach provides the optimal balance.]
)

The final output is a tuple $(H, "priors", "obs_flip")$ where:
- $H$: Parity check matrix of shape $("num_detectors", "num_hyperedges")$
- $"priors"$: Prior error probabilities per hyperedge
- $"obs_flip"$: Observable flip probabilities $P("obs flip" | "hyperedge fires")$

=== Step 3: Estimating Logical Error Rate

With the parity check matrix $H$ constructed, we can now decode syndrome samples and estimate the logical error rate.

==== Decoding Pipeline

The BP+OSD decoding pipeline consists of three stages:

#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // Boxes
    rect((0, 0), (3, 1.5), name: "bp")
    content("bp", [*BP Decoder*\ Marginal $P(e_j | bold(s))$])
    
    rect((4.5, 0), (7.5, 1.5), name: "osd")
    content("osd", [*OSD Post-Process*\ Hard solution $hat(bold(e))$])
    
    rect((9, 0), (12, 1.5), name: "xor")
    content("xor", [*XOR Chain*\ Predict $hat(L)$])
    
    // Arrows
    line((3, 0.75), (4.5, 0.75), mark: (end: ">"))
    line((7.5, 0.75), (9, 0.75), mark: (end: ">"))
    
    // Input/Output labels
    content((1.5, 2), [Syndrome $bold(s)$])
    line((1.5, 1.8), (1.5, 1.5), mark: (end: ">"))
    
    content((10.5, -0.7), [Prediction $hat(L) in {0, 1}$])
    line((10.5, -0.5), (10.5, 0), mark: (start: ">"))
  }),
  caption: [BP+OSD decoding pipeline: BP computes soft marginals, OSD finds a hard solution, XOR chain predicts observable.]
)

1. *BP Decoding*: Given syndrome $bold(s)$, run belief propagation on the factor graph to compute marginal probabilities $P(e_j = 1 | bold(s))$ for each hyperedge $j$.

2. *OSD Post-Processing*: Use Ordered Statistics Decoding to find a hard solution $hat(bold(e))$ satisfying $H hat(bold(e)) = bold(s)$, ordered by BP marginals.

3. *XOR Probability Chain*: Compute the predicted observable value using soft probabilities.

==== XOR Probability Chain for Observable Prediction

The key insight is that observable prediction must account for the *soft* flip probabilities stored in `obs_flip`. When hyperedges are merged, `obs_flip[j]` contains $P("obs flip" | "hyperedge " j " fires")$, not a binary indicator.

#theorem("XOR Probability Chain")[
  Given a solution $hat(bold(e))$ and observable flip probabilities $"obs_flip"$, the probability of an odd number of observable flips is computed iteratively:
  $ P_"flip" = P_"flip" dot (1 - "obs_flip"[j]) + "obs_flip"[j] dot (1 - P_"flip") $
  for each $j$ where $hat(e)_j = 1$. The predicted observable is $hat(L) = bb(1)[P_"flip" > 0.5]$.
]

The implementation:

```python
def compute_observable_predictions_batch(solutions, obs_flip):
    batch_size = solutions.shape[0]
    predictions = np.zeros(batch_size, dtype=int)
    
    for b in range(batch_size):
        p_flip = 0.0
        for i in np.where(solutions[b] == 1)[0]:
            # XOR probability: P(A XOR B) = P(A)(1-P(B)) + P(B)(1-P(A))
            p_flip = p_flip * (1 - obs_flip[i]) + obs_flip[i] * (1 - p_flip)
        predictions[b] = int(p_flip > 0.5)
    
    return predictions
```

#keypoint[
  If `merge_hyperedges=False`, then `obs_flip` contains binary values ${0, 1}$, and the XOR chain reduces to simple parity: $hat(L) = sum_j hat(e)_j dot "obs_flip"[j] mod 2$.
]

==== Logical Error Rate Estimation

The logical error rate (LER) is estimated by comparing predictions to ground truth:

$ "LER" = 1/N sum_(i=1)^N bb(1)[hat(L)^((i)) eq.not L^((i))] $

where $N$ is the number of syndrome samples, $hat(L)^((i))$ is the predicted observable for sample $i$, and $L^((i))$ is the ground truth.

==== Threshold Analysis

The *threshold* $p_"th"$ is the physical error rate below which increasing code distance reduces the logical error rate. For rotated surface codes with circuit-level depolarizing noise, the threshold is approximately *0.7%* (Bravyi et al., Nature 2024).

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (center, center, center, center),
    stroke: 0.5pt,
    inset: 8pt,
    [*Distance*], [*$p = 0.005$*], [*$p = 0.007$*], [*$p = 0.009$*],
    [$d = 3$], [$tilde 0.03$], [$tilde 0.06$], [$tilde 0.10$],
    [$d = 5$], [$tilde 0.01$], [$tilde 0.04$], [$tilde 0.09$],
    [$d = 7$], [$tilde 0.005$], [$tilde 0.03$], [$tilde 0.08$],
  ),
  caption: [Example logical error rates for BP+OSD decoder. Below threshold ($p < 0.007$), larger distances achieve lower LER. Above threshold, the trend reverses.]
)

At the threshold, curves for different distances *cross*: below threshold, larger $d$ gives lower LER; above threshold, larger $d$ gives *higher* LER due to more opportunities for errors to accumulate.

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

= Tropical Tensor Network

In this section, we introduce a complementary approach to decoding: *tropical tensor networks*. While BP+OSD performs approximate inference followed by algebraic post-processing, tropical tensor networks provide a framework for *exact* maximum a posteriori (MAP) inference by reformulating the problem in terms of tropical algebra.

The key insight is that finding the most probable error configuration corresponds to an optimization problem that can be solved exactly using tensor network contractions in the tropical semiring. This approach is particularly powerful for structured codes where the underlying factor graph has bounded treewidth.

== Tropical Semiring

#definition[
  The *tropical semiring* (also called the *max-plus algebra*) is the algebraic structure $(RR union {-infinity}, plus.circle, times.circle)$ where:
  - *Tropical addition*: $a plus.circle b = max(a, b)$
  - *Tropical multiplication*: $a times.circle b = a + b$ (ordinary addition)
  - *Additive identity*: $-infinity$ (since $max(a, -infinity) = a$)
  - *Multiplicative identity*: $0$ (since $a + 0 = a$)
]

#keypoint[
  The tropical semiring satisfies all semiring axioms:
  - Associativity: $(a plus.circle b) plus.circle c = a plus.circle (b plus.circle c)$
  - Commutativity: $a plus.circle b = b plus.circle a$
  - Distributivity: $a times.circle (b plus.circle c) = (a times.circle b) plus.circle (a times.circle c)$

  This algebraic structure allows us to replace standard summation with maximization while preserving the correctness of tensor contractions.
]

The tropical semiring was first systematically studied in the context of automata theory and formal languages @pin1998tropical. Its connection to optimization problems makes it particularly useful for decoding applications.

#figure(
  canvas({
    import draw: *

    // Standard vs Tropical comparison
    set-style(stroke: 0.8pt)

    // Left box: Standard algebra
    rect((-4.5, -1.5), (-0.5, 1.5), stroke: blue, radius: 4pt, fill: rgb("#f0f7ff"))
    content((-2.5, 1.1), text(weight: "bold", size: 9pt)[Standard Algebra])
    content((-2.5, 0.4), text(size: 8pt)[$a + b = "sum"$])
    content((-2.5, -0.1), text(size: 8pt)[$a times b = "product"$])
    content((-2.5, -0.7), text(size: 8pt)[Used for: Marginals])

    // Right box: Tropical algebra
    rect((0.5, -1.5), (4.5, 1.5), stroke: orange, radius: 4pt, fill: rgb("#fffaf0"))
    content((2.5, 1.1), text(weight: "bold", size: 9pt)[Tropical Algebra])
    content((2.5, 0.4), text(size: 8pt)[$a plus.circle b = max(a, b)$])
    content((2.5, -0.1), text(size: 8pt)[$a times.circle b = a + b$])
    content((2.5, -0.7), text(size: 8pt)[Used for: MAP/MPE])

    // Arrow
    line((-0.3, 0), (0.3, 0), stroke: 1.5pt, mark: (end: ">"))
  }),
  caption: [Standard algebra vs tropical algebra: switching the algebraic structure transforms marginalization into optimization]
)

== From Probabilistic Inference to Tropical Algebra

Recall that the MAP (Maximum A Posteriori) decoding problem seeks:
$ bold(e)^* = arg max_(bold(e) : H bold(e) = bold(s)) P(bold(e)) $

For independent bit-flip errors with probability $p$, the probability factors as:
$ P(bold(e)) = product_(i=1)^n P(e_i) = product_(i=1)^n p^(e_i) (1-p)^(1-e_i) $

Taking the logarithm transforms products into sums:
$ log P(bold(e)) = sum_(i=1)^n log P(e_i) = sum_(i=1)^n [e_i log p + (1-e_i) log(1-p)] $

#keypoint[
  In the log-probability domain:
  - *Products become sums*: $log(P dot Q) = log P + log Q$
  - *Maximization is preserved*: $arg max_x f(x) = arg max_x log f(x)$

  This means finding the MAP estimate for a function $product_f phi_f (bold(e)_f)$ is equivalent to:
  $ bold(e)^* = arg max_(bold(e) : H bold(e) = bold(s)) sum_f log phi_f (bold(e)_f) $
  where each factor $phi_f$ contributes additively in log-space.
]

The connection to tropical algebra becomes clear: if we replace standard tensor contractions (sum over products) with tropical contractions (max over sums), we transform marginal probability computation into MAP computation @pearl1988probabilistic.

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    stroke: 0.5pt,
    [*Operation*], [*Standard (Marginals)*], [*Tropical (MAP)*],
    [Combine factors], [$phi_a dot phi_b$], [$log phi_a + log phi_b$],
    [Eliminate variable], [$sum_x$], [$max_x$],
    [Result], [Partition function $Z$], [Max log-probability],
  ),
  caption: [Correspondence between standard and tropical tensor operations]
)

*Example:* Consider a simple Markov chain with three binary variables $x_1, x_2, x_3 in {0, 1}$ and two factors:

$ P(x_1, x_2, x_3) = phi_1(x_1, x_2) dot phi_2(x_2, x_3) $

#figure(
  canvas({
    import draw: *
    set-style(stroke: 0.8pt)

    // Factor graph at top
    content((0, 3.2), text(weight: "bold", size: 9pt)[Factor Graph])

    // Variable nodes (circles)
    circle((-1.5, 2.2), radius: 0.3, fill: white, name: "x1")
    content("x1", text(size: 8pt)[$x_1$])
    circle((0, 2.2), radius: 0.3, fill: white, name: "x2")
    content("x2", text(size: 8pt)[$x_2$])
    circle((1.5, 2.2), radius: 0.3, fill: white, name: "x3")
    content("x3", text(size: 8pt)[$x_3$])

    // Factor nodes (squares)
    rect((-0.95, 2.0), (-0.55, 2.4), fill: rgb("#e0e0e0"), name: "phi1")
    content("phi1", text(size: 6pt)[$phi_1$])
    rect((0.55, 2.0), (0.95, 2.4), fill: rgb("#e0e0e0"), name: "phi2")
    content("phi2", text(size: 6pt)[$phi_2$])

    // Factor graph edges
    line((-1.2, 2.2), (-0.95, 2.2))
    line((-0.55, 2.2), (-0.3, 2.2))
    line((0.3, 2.2), (0.55, 2.2))
    line((0.95, 2.2), (1.2, 2.2))

    // Arrows down to two paths
    line((-0.5, 1.6), (-2.5, 0.8), mark: (end: ">"))
    line((0.5, 1.6), (2.5, 0.8), mark: (end: ">"))

    // Left path: Standard algebra
    content((-2.5, 0.5), text(weight: "bold", size: 8pt)[Standard Algebra])
    rect((-3.8, -0.6), (-1.2, 0.2), stroke: 0.5pt, fill: rgb("#f0f7ff"))
    content((-2.5, -0.2), text(size: 8pt)[$Z = sum_(x_1,x_2,x_3) phi_1 dot phi_2$])

    // Right path: Tropical algebra
    content((2.5, 0.5), text(weight: "bold", size: 8pt)[Tropical Algebra])
    rect((1.2, -0.6), (3.8, 0.2), stroke: 0.5pt, fill: rgb("#fffaf0"))
    content((2.5, -0.2), text(size: 8pt)[$Z_"trop" = max_(x_1,x_2,x_3) (log phi_1 + log phi_2)$])

    // Operations labels
    content((-2.5, -0.9), text(size: 7pt, fill: gray)[sum + multiply])
    content((2.5, -0.9), text(size: 7pt, fill: gray)[max + add])

    // Arrows to results
    line((-2.5, -1.2), (-2.5, -1.6), mark: (end: ">"))
    line((2.5, -1.2), (2.5, -1.6), mark: (end: ">"))

    // Results
    content((-2.5, -1.9), text(size: 8pt)[Partition function $Z$])
    content((2.5, -1.9), text(size: 8pt)[Max log-probability])

    // Log transform arrow connecting the two
    line((-1.0, -1.9), (0.8, -1.9), stroke: (dash: "dashed"), mark: (end: ">"))
    content((0, -1.6), text(size: 6pt, fill: gray)[log transform])
  }),
  caption: [Standard vs tropical contraction of a Markov chain. The same factor graph structure supports both marginal computation (standard algebra) and MAP inference (tropical algebra).]
)

The partition function in standard algebra sums over all configurations:
$ Z = sum_(x_1, x_2, x_3) phi_1(x_1, x_2) dot phi_2(x_2, x_3) $

The same structure in tropical algebra computes the maximum log-probability:
$ Z_"trop" = max_(x_1, x_2, x_3) [log phi_1(x_1, x_2) + log phi_2(x_2, x_3)] $

#keypoint[
  *Beyond a Change of Language* @liu2021tropical: Tropical tensor networks provide computational capabilities unavailable in traditional approaches:

  + *Automatic Differentiation for Configuration Recovery*: Backpropagating through tropical contraction yields gradient "masks" that directly identify optimal variable assignments $bold(x)^*$---no separate search phase is needed.

  + *Degeneracy Counting via Mixed Algebras*: By tracking $(Z_"trop", n)$ where $n$ counts multiplicities, one simultaneously finds the optimal value AND counts all solutions achieving it in a single contraction pass.

  + *GPU-Accelerated Tropical BLAS*: Tropical matrix multiplication maps to highly optimized GPU kernels, enabling exact ground states for 1024-spin Ising models and 512-qubit D-Wave graphs in under 100 seconds.
]

== Tensor Network Representation

A tensor network represents the factorized probability dis
tribution as a graph where nodes of tensors correspond to factors $phi_f$ and the edges of  correspond to functions that contract the variables.

#definition[
  Given a factor graph with factors ${phi_f}$ and variables ${x_i}$, the corresponding *tensor network* consists of:
  - A tensor $T_f$ for each factor, with indices corresponding to the variables in $phi_f$
  - The *contraction* of the network computes: $sum_(x_1, ..., x_n) product_f T_f (bold(x)_f)$

  In the tropical semiring, this becomes: $max_(x_1, ..., x_n) sum_f T_f (bold(x)_f)$
]

The efficiency of tensor network contraction depends critically on the *contraction order*---the sequence in which variables are eliminated.

#keypoint[
  The *treewidth* of the factor graph determines the computational complexity:
  - A contraction order exists with complexity $O(n dot d^(w+1))$ where $w$ is the treewidth
  - For sparse graphs (like LDPC codes), treewidth can be small, enabling efficient exact inference
  - Tools like `omeco` find near-optimal contraction orders using greedy heuristics
]

#figure(
  canvas({
    import draw: *

    // Factor graph to tensor network illustration
    set-style(stroke: 0.8pt)

    // Title
    content((0, 2.3), text(weight: "bold", size: 9pt)[Factor Graph → Tensor Network])

    // Factor graph (left side)
    // Variable nodes
    circle((-3, 1), radius: 0.25, fill: white, name: "x1")
    content("x1", text(size: 7pt)[$x_1$])
    circle((-2, 1), radius: 0.25, fill: white, name: "x2")
    content("x2", text(size: 7pt)[$x_2$])
    circle((-1, 1), radius: 0.25, fill: white, name: "x3")
    content("x3", text(size: 7pt)[$x_3$])

    // Factor nodes
    rect((-3.2, -0.2), (-2.8, 0.2), fill: rgb("#e0e0e0"), name: "f1")
    content("f1", text(size: 6pt)[$phi_1$])
    rect((-2.2, -0.2), (-1.8, 0.2), fill: rgb("#e0e0e0"), name: "f2")
    content("f2", text(size: 6pt)[$phi_2$])
    rect((-1.2, -0.2), (-0.8, 0.2), fill: rgb("#e0e0e0"), name: "f12")
    content("f12", text(size: 6pt)[$phi_3$])

    // Edges
    line((-3, 0.75), (-3, 0.2))
    line((-2, 0.75), (-2, 0.2))
    line((-1, 0.75), (-1, 0.2))
    line((-3, 0.75), (-1.2, 0.2))
    line((-2, 0.75), (-0.8, 0.2))

    // Arrow
    line((0, 0.5), (0.8, 0.5), stroke: 1.5pt, mark: (end: ">"))

    // Tensor network (right side)
    circle((2, 1), radius: 0.3, fill: rgb("#e0e0e0"), name: "t1")
    content("t1", text(size: 6pt)[$T_1$])
    circle((3, 1), radius: 0.3, fill: rgb("#e0e0e0"), name: "t2")
    content("t2", text(size: 6pt)[$T_2$])
    circle((2.5, 0), radius: 0.3, fill: rgb("#e0e0e0"), name: "t3")
    content("t3", text(size: 6pt)[$T_3$])

    // Tensor edges (contracted indices)
    line((2.3, 0.85), (2.35, 0.28), stroke: 1pt + blue)
    line((2.7, 0.85), (2.65, 0.28), stroke: 1pt + blue)

    // Open edges (free indices)
    line((1.7, 1), (1.3, 1), stroke: 1pt)
    line((3.3, 1), (3.7, 1), stroke: 1pt)
    line((2.5, -0.3), (2.5, -0.6), stroke: 1pt)
  }),
  caption: [Factor graph representation as a tensor network. Edges between tensors represent indices to be contracted (summed/maximized over).]
)

The contraction process proceeds by repeatedly selecting a variable to eliminate:

```python
# Conceptual contraction loop (simplified)
for var in elimination_order:
    bucket = [tensor for tensor in tensors if var in tensor.indices]
    combined = tropical_contract(bucket, eliminate=var)
    tensors.update(combined)
```

== Backpointer Tracking for MPE Recovery

A critical challenge with tensor network contraction is that it only computes the *value* of the optimal solution (the maximum log-probability), not the *assignment* that achieves it.

#definition[
  A *backpointer* is a data structure that records, for each $max$ operation during contraction:
  - The indices of eliminated variables
  - The $arg max$ value for each output configuration

  Formally, when computing $max_x T(y, x)$, we store: $"bp"(y) = arg max_x T(y, x)$
]

The recovery algorithm traverses the contraction tree in reverse:

#figure(
  canvas({
    import draw: *

    set-style(stroke: 0.8pt)

    // Contraction tree
    content((0, 3), text(weight: "bold", size: 9pt)[Contraction Tree with Backpointers])

    // Root
    circle((0, 2), radius: 0.35, fill: rgb("#90EE90"), name: "root")
    content("root", text(size: 7pt)[root])

    // Level 1
    circle((-1.5, 0.8), radius: 0.35, fill: rgb("#ADD8E6"), name: "n1")
    content("n1", text(size: 7pt)[$C_1$])
    circle((1.5, 0.8), radius: 0.35, fill: rgb("#ADD8E6"), name: "n2")
    content("n2", text(size: 7pt)[$C_2$])

    // Level 2 (leaves)
    circle((-2.2, -0.4), radius: 0.3, fill: rgb("#FFE4B5"), name: "l1")
    content("l1", text(size: 6pt)[$T_1$])
    circle((-0.8, -0.4), radius: 0.3, fill: rgb("#FFE4B5"), name: "l2")
    content("l2", text(size: 6pt)[$T_2$])
    circle((0.8, -0.4), radius: 0.3, fill: rgb("#FFE4B5"), name: "l3")
    content("l3", text(size: 6pt)[$T_3$])
    circle((2.2, -0.4), radius: 0.3, fill: rgb("#FFE4B5"), name: "l4")
    content("l4", text(size: 6pt)[$T_4$])

    // Edges with backpointer annotations
    line((0, 1.65), (-1.2, 1.1), stroke: 1pt)
    line((0, 1.65), (1.2, 1.1), stroke: 1pt)
    line((-1.5, 0.45), (-2, -0.1), stroke: 1pt)
    line((-1.5, 0.45), (-1, -0.1), stroke: 1pt)
    line((1.5, 0.45), (1, -0.1), stroke: 1pt)
    line((1.5, 0.45), (2, -0.1), stroke: 1pt)

    // Backpointer arrows (dashed, showing recovery direction)
    line((0.3, 2), (1.2, 1.15), stroke: (dash: "dashed", paint: red), mark: (end: ">"))
    content((1.1, 1.7), text(size: 6pt, fill: red)[bp])

    line((-0.3, 2), (-1.2, 1.15), stroke: (dash: "dashed", paint: red), mark: (end: ">"))
    content((-1.1, 1.7), text(size: 6pt, fill: red)[bp])
  }),
  caption: [Contraction tree with backpointers. During contraction (bottom-up), backpointers record argmax indices. During recovery (top-down, dashed arrows), backpointers are traced to reconstruct the optimal assignment.]
)

The implementation in the `tropical_in_new/` module demonstrates this pattern:

```python
# From tropical_in_new/src/primitives.py
@dataclass
class Backpointer:
    """Stores argmax metadata for eliminated variables."""
    elim_vars: Tuple[int, ...]      # Which variables were eliminated
    elim_shape: Tuple[int, ...]     # Domain sizes
    out_vars: Tuple[int, ...]       # Remaining output variables
    argmax_flat: torch.Tensor       # Flattened argmax indices

def tropical_reduce_max(tensor, vars, elim_vars, track_argmax=True):
    """Tropical max-reduction with optional backpointer tracking."""
    # ... reshape tensor to separate kept and eliminated dimensions ...
    values, argmax_flat = torch.max(flat, dim=-1)
    if track_argmax:
        backpointer = Backpointer(elim_vars, elim_shape, out_vars, argmax_flat)
    return values, backpointer
```

The recovery algorithm traverses the tree from root to leaves:

```python
# From tropical_in_new/src/mpe.py
def recover_mpe_assignment(root) -> Dict[int, int]:
    """Recover MPE assignment from a contraction tree with backpointers."""
    assignment: Dict[int, int] = {}

    def traverse(node, out_assignment):
        assignment.update(out_assignment)
        if isinstance(node, ReduceNode):
            # Use backpointer to recover eliminated variable values
            elim_assignment = argmax_trace(node.backpointer, out_assignment)
            child_assignment = {**out_assignment, **elim_assignment}
            traverse(node.child, child_assignment)
        elif isinstance(node, ContractNode):
            # Propagate to both children
            elim_assignment = argmax_trace(node.backpointer, out_assignment)
            combined = {**out_assignment, **elim_assignment}
            traverse(node.left, {v: combined[v] for v in node.left.vars})
            traverse(node.right, {v: combined[v] for v in node.right.vars})

    # Start from root with initial assignment from final tensor
    initial = unravel_argmax(root.values, root.vars)
    traverse(root, initial)
    return assignment
```

== Application to Error Correction Decoding

For quantum error correction, the MAP decoding problem is:
$ bold(e)^* = arg max_(bold(e) : H bold(e) = bold(s)) P(bold(e)) $

The syndrome constraint $H bold(e) = bold(s)$ can be incorporated as hard constraints (factors that are $-infinity$ for invalid configurations and $0$ otherwise) @farrelly2020parallel.

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    stroke: 0.5pt,
    [*Aspect*], [*BP+OSD*], [*Tropical TN*],
    [Inference type], [Approximate marginals], [Exact MAP],
    [Degeneracy handling], [OSD post-processing], [Naturally finds one optimal],
    [Output], [Soft decisions → hard], [Direct hard assignment],
    [Complexity], [$O(n^3)$ for OSD], [Exp. in treewidth],
    [Parallelism], [Iterative], [Highly parallelizable],
  ),
  caption: [Comparison of BP+OSD and tropical tensor network decoding approaches]
)

#keypoint[
  *Advantages of tropical tensor networks for decoding:*
  - *Exactness*: Guaranteed to find the MAP solution (no local minima)
  - *No iterations*: Single forward pass plus backtracking
  - *Natural for structured codes*: Exploits graph structure via contraction ordering

  *Limitations:*
  - Complexity grows exponentially with treewidth
  - For dense or high-treewidth codes, may be less efficient than BP+OSD
  - Requires careful implementation of backpointer tracking
]

The tensor network approach is particularly well-suited to codes with local structure, such as topological codes where the treewidth grows slowly with system size @orus2019tensor.

== Complexity Considerations

The computational complexity of tropical tensor network contraction is governed by the *treewidth* of the underlying factor graph.

#definition[
  The *treewidth* $w$ of a graph is the minimum width of any tree decomposition, where width is one less than the size of the largest bag. Intuitively, it measures how "tree-like" the graph is.
]

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    stroke: 0.5pt,
    [*Code Type*], [*Treewidth*], [*Contraction Complexity*],
    [1D repetition], [$O(1)$], [$O(n)$],
    [2D toric], [$O(sqrt(n))$], [$O(n dot 2^(sqrt(n)))$],
    [LDPC (sparse)], [$O(log n)$ to $O(sqrt(n))$], [Varies],
    [Dense codes], [$O(n)$], [$O(2^n)$ -- intractable],
  ),
  caption: [Treewidth and complexity for different code families]
)

#keypoint[
  For LDPC codes used in quantum error correction:
  - The sparse parity check matrix leads to bounded-degree factor graphs
  - Greedy contraction order heuristics (like those in `omeco`) often find good orderings
  - The practical complexity is often much better than worst-case bounds suggest

  The tropical tensor network approach provides a systematic way to exploit code structure for efficient exact decoding when the treewidth permits.
]

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

