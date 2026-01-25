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
  leading: 0.8em,
  first-line-indent: 2em
)

#show heading: it => [
  #v(0.5em)
  #text(weight: "bold", size: 1.15em, it)
  #v(0.5em)
]

// Helper macros for common quantum mechanics notation
#let ket(x) = $|#x >$
#let bra(x) = $> #x|$
#let braket(x, y) = $> #x | #y >$

#import "@preview/qec-thrust:0.1.1": *

#let note(title, body) = {
  block(
    fill: luma(90%),
    stroke: (left: 4pt + orange),
    inset: 12pt,
    radius: (right: 4pt),
    width: 100%,
    [
      #text(weight: "bold", fill: orange.darken(20%), size: 1.1em)[#title]
      #v(0.5em)
      #body
    ]
  )
}

#align(center)[
  #text(size: 2em, weight: "bold")[
    Quantum Error Correction and BP Decoding Notes
  ]

  #v(0.8em)

  #text(size: 1.1em)[
    Shen Yang
  ]

  #text(size: 1em, fill: gray.darken(30%))[
    School of Electronic Science and Engineering, Southeast University
  ]
]

#v(2em)
#outline(
  title: [Contents],   // ToC title
  depth: 3,        // depth (1~6)
  indent: 1.5em      // child indent
)

#pagebreak()


= Part I: Fundamentals of Quantum Coding

== 1. Why do we need quantum coding?

The power of quantum computers comes from superposition and entanglement, but that also makes them extremely fragile. Unlike #link(<quantum-storage>)[#text(fill:red)[classical computers]], quantum systems face two core challenges: <quantum-storage-back>

+ *Decoherence and environmental noise*: quantum states interact uncontrollably with the environment, causing phase flips (Phase Flip, Z error) or bit flips (Bit Flip, X error).
+ *No-Cloning Theorem*: we cannot simply copy quantum states like classical error-correcting codes (e.g., repetition code $0 arrow 000$) to do majority voting.

Therefore, the core idea of quantum error correction (QEC) is: *hide quantum information inside an entangled subspace of a many-body system*. We do not copy states; instead, we spread the information of a single logical qubit across many physical qubits via **nonlocal correlations**.

== 2. Encoding principle: stabilizer formalism

Before introducing specific codes (e.g., LDPC and surface codes), we must understand **stabilizers**, which are the basis for BP (belief propagation) decoding.

For $n$ physical qubits, the state space is a $2^n$-dimensional Hilbert space. We define a logical subspace (code space) $cal(C)$ such that all logical states $| psi_L > in cal(C)$ satisfy a set of eigenvalue equations:

$
  S_i | psi_L > = +1 | psi_L >, quad forall i
$

These operators $S_i$ are called **stabilizers**.
- *No error*: measuring $S_i$ always yields $+1$ (or binary 0).
- *With error*: if an error $E$ occurs and $E$ anticommutes with $S_i$ ($E S_i = - S_i E$), measuring $S_i$ yields $-1$ (binary 1).

By detecting which stabilizers "fire" (value -1), we can infer the error without directly measuring the quantum state (thus avoiding destroying superposition).

== 3. Quantum low-density parity-check codes (QLDPC)

Quantum LDPC codes generalize classical LDPC codes to the quantum domain. Their sparsity makes graph-based decoding (like BP) possible.

=== 3.1 CSS construction (Calderbank-Shor-Steane)
To handle quantum errors (X and Z), the most common construction is the CSS code. It separates correction into two independent parts. We need two classical parity-check matrices $H_X$ and $H_Z$.

- $H_X$: detects Z errors (rows represent X-type stabilizers).
- $H_Z$: detects X errors (rows represent Z-type stabilizers).

*Constraint*: to #link(<Commute>)[#text(fill:red)[make stabilizers commute]] (commute), these matrices must satisfy the orthogonality constraint: <Commute-back>
$
  H_X dot H_Z^T = 0 mod 2
$

=== 3.2 Factor-graph representation (Tanner graph)
LDPC codes can be represented by factor graphs. For BP, there are two types of nodes:
- *Variable nodes*: physical data qubits.
- *Check nodes*: stabilizers (X checks or Z checks).

If $H_X$ and $H_Z^T$ are sparse (each row and column has O(1) ones) and satisfy
$
  H_X dot H_Z^T = 0 mod 2
$
then the stabilizer code is called a quantum LDPC (QLDPC) code.

Below is a typical Tanner graph example. You can see the "locality": each check node connects to only a few variable nodes.

#figure(
  caption: [Tanner graph (factor graph) representation of a quantum error-correcting code],
  kind: "image",
  supplement: "Figure",
  block(
    height: 180pt,
    width: 100%,
    stroke: 0.5pt + gray.lighten(60%),
    radius: 5pt,
    inset: 10pt,
    align(center + horizon)[
      #box(width: 300pt, height: 140pt)[
        // Coordinate parameters
        #let vy = 20pt   // variable node y
        #let cy = 120pt  // check node y
        
        // Variable node x
        #let v1x = 50pt
        #let v2x = 110pt
        #let v3x = 170pt
        #let v4x = 230pt
        #let v5x = 290pt
        
        // Check node x
        #let c1x = 80pt
        #let c2x = 170pt
        #let c3x = 260pt

        // 1. Edges (H-matrix ones)
        // Bottom layer to avoid covering nodes
        
        // Check 1 connects v1, v2, v3
        #place(line(start: (c1x, cy), end: (v1x, vy), stroke: 1.5pt + gray))
        #place(line(start: (c1x, cy), end: (v2x, vy), stroke: 1.5pt + gray))
        #place(line(start: (c1x, cy), end: (v3x, vy), stroke: 1.5pt + gray))
        
        // Check 2 connects v2, v4
        #place(line(start: (c2x, cy), end: (v2x, vy), stroke: 1.5pt + gray))
        #place(line(start: (c2x, cy), end: (v4x, vy), stroke: 1.5pt + gray))

        // Check 3 connects v3, v4, v5
        #place(line(start: (c3x, cy), end: (v3x, vy), stroke: 1.5pt + gray))
        #place(line(start: (c3x, cy), end: (v4x, vy), stroke: 1.5pt + gray))
        #place(line(start: (c3x, cy), end: (v5x, vy), stroke: 1.5pt + gray))

        // 2. Variable nodes (physical qubits)
        #let draw_vnode(x, label) = {
          place(dx: x - 10pt, dy: vy - 10pt, circle(radius: 10pt, fill: white, stroke: 1.5pt + black))
          place(dx: x - 5pt, dy: vy - 25pt, text(size: 10pt)[#label])
        }
        
        #draw_vnode(v1x, $d_1$)
        #draw_vnode(v2x, $d_2$)
        #draw_vnode(v3x, $d_3$)
        #draw_vnode(v4x, $d_4$)
        #draw_vnode(v5x, $d_5$)

        // 3. Check nodes (stabilizers)
        #let draw_cnode(x, label) = {
          place(dx: x - 10pt, dy: cy - 10pt, rect(width: 20pt, height: 20pt, fill: luma(230), stroke: 1.5pt + black))
          place(dx: x - 5pt, dy: cy + 15pt, text(size: 10pt, weight: "bold")[#label])
        }

        #draw_cnode(c1x, $S_1$)
        #draw_cnode(c2x, $S_2$)
        #draw_cnode(c3x, $S_3$)

        // 4. Labels
        #place(dx: 0pt, dy: vy - 5pt, text(size: 8pt, fill: gray, style: "italic")[Data qubits])
        #place(dx: 0pt, dy: cy + 5pt, text(size: 8pt, fill: gray, style: "italic")[Check operators])
      ]
    ]
  )
)

=== Diagram notes
- **Circle nodes ($d_i$)**: physical data qubits.
- **Square nodes ($S_j$)**: stabilizer check operators.
- **Edges**: if the $j$-th row and $i$-th column of $H$ is 1 (stabilizer $S_j$ involves qubit $d_i$), draw a line.

In the upcoming BP algorithm, information (probabilities/confidence) will pass along these **black edges** between variable and check nodes until convergence or the max iteration count.

== 4. Surface code

The surface code is a special QLDPC code defined on a 2D lattice. It is one of the most promising candidates for fault-tolerant quantum computing because its check operators are **local** (only nearby qubits).

=== 4.1 Geometry
We define physical qubits and stabilizers on a 2D grid.
- *Data qubits*: on edges (or vertices, depending on definition; here we adopt a Kitaev toric-code style for intuition).
- *Z stabilizers (plaquettes)*: on faces; detect Z-operator products.
- *X stabilizers (vertex/star)*: on vertices; detect X-operator products.

=== 4.2 Diagram explanation

Below is a surface-code diagram with code distance $d = 3$.

- **Black dots**: data qubits. There are 9 black dots, corresponding to 9 physical qubits. They store the quantum state and can suffer physical errors. They are not independent; together they encode one logical qubit.

- **Colored squares**: stabilizer operator regions. Each square corresponds to a stabilizer whose support is the four corner data qubits. Different colors distinguish $X$ and $Z$ stabilizers, interleaved to ensure mutual commutation and simultaneous measurement.

- **Half-circles on boundaries**: boundary stabilizers. With open boundaries, boundary stabilizers act on fewer than four data qubits. The half-circle indicates this incomplete structure.

In this structure, logical operators correspond to nontrivial operator chains across the lattice. For $d = 3$, any shortest chain from one boundary to the opposite boundary contains at least 3 data qubits, capturing the meaning of distance $d = 3$.

#figure(
  caption: [Surface code diagram with code distance $d = 3$],
  kind: "image",
  supplement: "Figure",
)[
  #canvas({
    import draw: *

    let d = 3

    surface-code((0, 0), size: 1.5, d, d, name: "sc-d3")

    for i in range(d) {
      for j in range(d) {
        content(
          (rel: (0.3, 0.3), to: "sc-d3" + "-" + str(i) + "-" + str(j)),
          [#(i * d + j + 1)],
        )
      }
    }
  })
]

=== 4.3 Detailed construction and stabilizer definitions

Based on the diagram, we can write the surface code precisely. Its power comes from **topology**: information depends on global lattice topology, not individual qubits.

==== 4.3.1 Physical layer: parity checks
In the diagram, we saw two types of stabilizers responsible for different error types.

- **Z stabilizer (plaquette operator)**
  Corresponds to the red square $Z_p$. It acts on the four data qubits around the face ($d_1, d_2, d_3, d_4$).
  Its definition:
  $
    S_p^Z = Z_(d 1) Z_(d 2) Z_(d 3) Z_(d 4)
  $
  *Function*: detects **bit-flip errors (X error)**.
  If $d_1$ has an $X$ error, since $X Z = - Z X$ (anticommute), measuring $S_p^Z$ yields eigenvalue $-1$. We say the plaquette is "excited" (a defect is detected).

- **X stabilizer (vertex operator)**
  Corresponds to the blue dot $X_v$. It acts on the four data qubits touching the vertex.
  For example, the top-right vertex $X_(v 2)$ connects $d_1, d_4, d_5$ and an unshown qubit above.
  Its definition:
  $
    S_v^X = product_(i in "star"(v)) X_i
  $
  *Function*: detects **phase-flip errors (Z error)**.
  If a neighboring qubit has a $Z$ error, the vertex measurement becomes $-1$.

==== 4.3.2 Logical layer: logical qubits

If physical qubits are used for checks, where is the "real" information stored?
Answer: **operator chains that span the entire lattice**.

For a $d × d$ lattice (code distance $d$):

- **Logical $overline(Z)_L$ operator**:  
  a chain of $Z$ operators from the **top boundary** to the **bottom boundary**.
  $
    overline(Z)_L = Z_1 · Z_k · … · Z_m  ("vertical span")
  $
  This chain intersects each X stabilizer either 0 or 2 times, so it commutes with all stabilizers.

- **Logical $overline(X)_L$ operator**:  
  a chain of $X$ operators from the **left boundary** to the **right boundary**.
  $
    overline(X)_L = X_a · X_b · … · X_c  ("horizontal span")
  $
  This chain commutes with all Z stabilizers (plaquettes).

#note("Key point")[  
$overline(X)_L$ and $overline(Z)_L$ must
**intersect at an odd number (typically 1) of physical qubits**.
Because $X$ and $Z$ anticommute at that location, the logical operators satisfy:

$
  overline(X)_L · overline(Z)_L
  = - overline(Z)_L · overline(X)_L
$

This is the required algebra for a logical qubit.]

==== 4.3.3 Code distance (Code Distance $d$)
*Definition*: the minimum number of physical qubit operations required to transform one logical state (e.g. $|overline(0) >$) into an orthogonal logical state (e.g. $|overline(1) >$).
Equivalently, it is the minimum weight of a nontrivial logical operator.

#note("Why does d determine error-correction ability?")[
We have a golden formula:
$ d = 2t + 1 $
where $t$ is the number of errors we can guarantee to correct.

- *Intuition*:
  Imagine $|overline(0) >$ and $|overline(1) >$ as two points separated by distance $d$.
  - If $t$ errors occur, the state is pushed $t$ steps away from $|overline(0) >$.
  - As long as $t < d/2$, we are still closer to $|overline(0) >$ than to $|overline(1) >$. The decoder pulls us back to $|overline(0) >$ by a nearest-neighbor rule.
  - Once errors reach $d/2$ or more (e.g. $(d+1)/2$), we cross the midpoint and are closer to $|overline(1) >$. The decoder "corrects" toward $|overline(1) >$, causing a logical error.]

#note("Geometric meaning in surface codes")[
On the surface-code lattice, $d$ is the **linear size** of the grid.
- To cause a logical flip, an error chain must **span** the entire lattice (left-right or top-bottom).
- Therefore $d$ is the shortest path length from one boundary to the opposite.
- Increasing grid size increases $d$, exponentially lowering logical error rate $P_L tilde (p_"phys"/p_"th")^(d/2)$.]

=== 4.4 Error detection

Assume an **$X$ error** occurs at **$d_1$**:

1. **Physical layer**: the state of $d_1$ flips.
2. **Check layer**:
   - The **red plaquette ($Z_p$)** containing $d_1$ anticommutes, measurement becomes $-1$.
   - The **adjacent upper plaquette** (not shown) also becomes $-1$.
   - All other plaquettes and vertices are unaffected (X errors commute with X stabilizers).
3. **Syndrome**: we see two adjacent "excited plaquettes" in space.
4. **Decoder task**: the decoder (e.g., BP) infers that "the most likely explanation is an error on the shared edge (i.e. $d_1$) connecting the two red plaquettes."

This is exactly how BP works: infer errors (causes) from syndromes (results).

= Part II: Belief Propagation (BP) decoding

Belief propagation (BP), also called the sum-product algorithm, is an **iterative message-passing** algorithm on factor graphs.

In QEC, our goal is: given the observed syndrome vector $s$ (which stabilizers fired), compute the marginal probability $P(e_i | s)$ that each physical qubit $i$ has an error $e_i$.

Direct marginalization is exponential (NP-hard). BP approximates this via local message passing. For numerical stability and to convert products to sums, we use the **log-likelihood ratio (LLR)**.

== 1. Core definition: LLR

For each variable (physical qubit), define the LLR as the natural log of the ratio of "no error" to "error":

$
  L = ln (P(x=0) / P(x=1))
$

- $L > 0$: tends to believe no error (0).
- $L < 0$: tends to believe error (1).
- Larger $|L|$: higher confidence.

== 2. Algorithm flow

BP has two message directions: variable to check $(v arrow c)$ and check to variable $(c arrow v)$.

=== Step 0: Initialization

Based on the physical channel error rate #link(<Prior>)[#text(fill:red)[$p$ (prior probability)]], initialize each data qubit $v_i$'s LLR.
Assume each qubit has error probability $p$: <Prior-back>

$
  L_i^((0)) = ln ((1-p)/p)
$
This is the algorithm's initial belief.

=== Step 1: Check node update ($c arrow v$)

*Intuition*: check node $c_j$ tells variable $v_i$: "Based on the other variables $v_k$ I connect to, and my syndrome $s_j$ (0 or 1), I think your value should be..."

This is a **parity-check** constraint. If the sum of other errors is even and $s_j=0$, then $v_i$ must be 0 (even), and so on.

Using properties of $tanh$, the message $R_(j arrow i)$ is:

$
  R_(j arrow i) = 2 tanh^(-1) ( (-1)^(s_j) product_(k in N(j) backslash i) tanh(Q_(k arrow j) / 2) )
$

- $N(j) backslash i$: all variables connected to check $j$, **excluding** $i$ (avoid feedback).
- $(-1)^(s_j)$: key term. If $s_j=1$ (check fired), it flips the sign and says "someone among you is wrong."
- $Q_(k arrow j)$: message from variable $k$ in the previous round.

=== Step 2: Variable node update ($v arrow c$)

*Intuition*: variable node $v_i$ aggregates messages from all neighboring checks and tells check $c_j$: "Combining channel prior and other checks, I believe my state is..."

With LLRs, products become **sums**:

$
  Q_(i arrow j) = L_i^((0)) + sum_(k in M(i) backslash j) R_(k arrow i)
$

- $L_i^((0))$: initial channel LLR.
- $M(i) backslash j$: all checks connected to $i$, **excluding** $j$.

=== Step 3: Decision and termination

After some iterations (alternating steps 1 and 2), or after reaching the max iterations, compute each bit's **total posterior LLR**:

$
  L_i^("total") = L_i^((0)) + sum_(k in M(i)) R_(k arrow i)
$
(Note: this sum includes all neighbors.)

**Hard decision**:
- If $L_i^("total") > 0 arrow hat(e)_i = 0$ (no error).
- If $L_i^("total") < 0 arrow hat(e)_i = 1$ (error).

Finally, check whether the inferred error vector $hat(e)$ satisfies the syndrome:
$
H · hat(e)^T = s^T,
$
which determines whether the current estimate matches the syndrome constraints.

If equal, decoding succeeds; otherwise decoding fails (or proceeds to post-processing like OSD).

== 3. Visual intuition: message passing

To help understand information flow, here is a simple local message-flow diagram.

#figure(
  caption: [BP message-passing: information exchange diagram],
  kind: "image",
  supplement: "Figure",
  block(
    height: 140pt,
    width: 100%,
    stroke: 0.5pt + gray.lighten(60%),
    radius: 5pt,
    inset: 10pt,
    align(center + horizon)[
      #box(width: 340pt, height: 120pt)[
        
        // Coordinates
        #let vy = 60pt      // baseline y
        #let v1x = 60pt     // left variable x
        #let cx = 170pt     // middle check x
        #let v2x = 280pt    // right variable x (neighbor)
        
        // 1. Structural edges (gray solid)
        #place(line(start: (v1x, vy), end: (cx, vy), stroke: 2pt + gray.lighten(70%)))
        #place(line(start: (cx, vy), end: (v2x, vy), stroke: 2pt + gray.lighten(70%)))

        // 2. Nodes
        // Left variable
        #place(dx: v1x - 12pt, dy: vy - 12pt, circle(radius: 12pt, fill: white, stroke: 1.5pt + black))
        #place(dx: v1x - 25pt, dy: vy - 35pt, text(weight: "bold")[$V_i$])
        
        // Middle check
        #place(dx: cx - 12pt, dy: vy - 12pt, rect(width: 24pt, height: 24pt, fill: luma(230), stroke: 1.5pt + black))
        #place(dx: cx - 5pt, dy: vy - 35pt, text(weight: "bold")[$C_j$])

        // Right variable
        #place(dx: v2x - 12pt, dy: vy - 12pt, circle(radius: 12pt, fill: white, stroke: 1.5pt + black))
        #place(dx: v2x - 10pt, dy: vy - 35pt, text(fill: gray)[$V_k$])

        // 3. Message flows
        
        // Message Q: variable -> check (blue, above, right)
        #let q_start = v1x + 15pt
        #let q_end = cx - 15pt
        #let q_y = vy - 8pt
        
        #place(line(start: (q_start, q_y), end: (q_end, q_y), stroke: (thickness: 1.5pt, paint: blue, dash: "dashed")))
        // Blue arrow
        #place(dx: q_end - 2pt, dy: q_y, polygon(fill: blue, (-4pt, 3pt), (-4pt, -3pt), (2pt, 0pt)))
        #place(dx: 95pt, dy: vy - 25pt, text(size: 9pt, fill: blue)[$Q_(i -> j)$])
        
        // Message R: check -> variable (red, below, left)
        #let r_start = cx - 15pt
        #let r_end = v1x + 15pt
        #let r_y = vy + 8pt
        
        #place(line(start: (r_start, r_y), end: (r_end, r_y), stroke: (thickness: 1.5pt, paint: red)))
        // Red arrow
        #place(dx: r_end + 2pt, dy: r_y, polygon(fill: red, (4pt, 3pt), (4pt, -3pt), (-2pt, 0pt)))
        #place(dx: 95pt, dy: vy + 15pt, text(size: 9pt, fill: red)[$R_(j -> i)$])

        // Additional input from the right neighbor (gray, left)
        #let n_start = v2x - 15pt
        #let n_end = cx + 15pt
        
        #place(line(start: (n_start, vy), end: (n_end, vy), stroke: (thickness: 1.5pt, paint: gray, dash: "dotted")))
        // Gray arrow
        #place(dx: n_end + 2pt, dy: vy, polygon(fill: gray, (4pt, 3pt), (4pt, -3pt), (-2pt, 0pt)))
        #place(dx: 215pt, dy: vy - 15pt, text(size: 8pt, fill: gray)[from neighbor $V_k$])

        // 4. Bottom note
        #place(dx: 120pt, dy: 100pt, 
          block(stroke: (left: 2pt+red), inset: 5pt, radius: 2pt, fill: luma(250))[
            #text(size: 9pt)[$R_(j -> i)$ depends on $sum V_k$]
          ]
        )
      ]
    ]
  )
)

This diagram shows BP message passing on a Tanner graph: variable nodes and check nodes exchange "information" along edges to approximate posterior probabilities.

- **Node meanings**:
  - Circle nodes $V_i, V_k$: variable nodes (a bit or error variable; in QEC, a physical qubit error random variable).
  - Square nodes $C_j$: check nodes (parity constraints; in quantum codes, stabilizer/syndrome constraints).

- **Gray solid structural edges**: Tanner graph edges (defined by ones in $H$). BP messages travel only along these edges.

- **Blue dashed message $Q_{(i→j)}$ (variable → check)**:  
  the belief/probability message from $V_i$ to $C_j$.  
  Key: it is a **local estimate of $V_i$**, excluding feedback from the same edge (do not use the message just received from $C_j$). Thus:
  $Q_{i→j}$ equals $V_i$'s prior (channel info) plus messages from other checks (excluding $C_j$).

- **Red solid message $R_{(j→i)}$ (check → variable)**:  
  the constraint feedback from $C_j$ to $V_i$ based on the parity condition.  
  It depends on other variables connected to $C_j$ (shown as "from neighbor $V_k$"), so:
  - $R_{j→i}$ aggregates all $Q_{k→j}$ (for $k≠i$),
  - combines with the syndrome/parity,
  - gives a consistency constraint on $V_i$.

- **Gray dotted line (from neighbor $V_k$)**:  
  emphasizes that $C_j$ needs inputs from other neighbors when computing $R_{j→i}$.

- **Overall iteration**:  
  BP iteratively updates $Q$ and $R$ so each variable node refines its posterior estimate. When messages converge or max iterations is reached, we make final decisions (hard/soft).

(Note: in CSS/QLDPC decoding, the same message-passing structure runs on the factor graphs of $H_X$ and $H_Z$, with syndromes as constraints.)


== 4. The quantum BP dilemma: degeneracy

While BP performs well for classical LDPC codes, in quantum error correction it faces a major issue: **degeneracy**.

In classical decoding, we must find the exact error. In quantum decoding, many different errors can correspond to the same logical state (e.g., $E$ and $E' = S_i E$ are equivalent).

BP tends to find the "most likely specific error" rather than the "most likely error class." This can cause hesitation or non-convergence at low signal-to-noise ratios. That is why we need **OSD (Ordered Statistics Decoding)** as a post-processing step to break the tie.

= Part III: Ordered Statistics Decoding (OSD)

BP outputs a "probability distribution" (soft information). In the following cases, BP alone is insufficient:
1.  **Non-convergence/oscillation**: with many short cycles, BP may loop and never settle.
2.  **Invalid syndrome**: a hard decision ($L>0 arrow 0$) can yield an error vector $e$ that does not satisfy $H e^T = s^T$.

**OSD (Ordered Statistics Decoding)** is a post-processing algorithm combining soft decisions with linear algebra. Core idea: **rather than guessing blindly, trust the "most confident" bits and solve for the rest.**

== 1. Core steps

Assume BP outputs the final LLR vector $L = (L_1, L_2, ..., L_n)$.

=== Step 1: Ordering
Sort all physical bits by absolute LLR magnitude $|L_i|$ in **descending** order.
- Larger $|L_i|$ means BP is more confident (either error or no error).
- Smaller $|L_i|$ (near 0) means ambiguous and least reliable.

Reorder the columns of $H$ accordingly to get a new matrix $H'$:
$
  H' = [ H_A | H_B ]
$
- $H_A$: columns for **high-confidence** bits.
- $H_B$: columns for **low-confidence** bits.

=== Step 2: Gaussian elimination and basis selection
We want an error vector $e$ that satisfies the syndrome $s$:
$H e^T = s^T$.

Since $H$ is usually wide (more variables than equations), there are infinitely many solutions. OSD's strategy: **let the high-confidence bits decide first.**

We perform **Gaussian elimination** on $H'$ and select linearly independent columns in $H_A$ as an **information set** (basis).
In practice, we try to reduce $H'$ to near-identity form:
$
  tilde(H) = [ I | P ]
$
This often involves further column swaps to ensure basis vectors come from the most confident positions.

=== Step 3: Solving
Once the basis is chosen (the most reliable positions), we "lock" those values (from BP hard decisions), and solve for the remaining positions uniquely so that $H e^T = s$ holds exactly.

== 2. OSD order (OSD-0 vs OSD-k)

The above is **OSD-0** (order-0), a greedy algorithm that fully trusts the ordering.
But BP confidence can be wrong.

So we introduce **OSD-k**:
- **Idea**: before solving, also try flipping the "least reliable among the basis" bits.
- **Process**:
  1. Choose the top $k$ bits that are in the basis but have the lowest confidence among the basis.
  2. Enumerate all $2^k$ flip combinations.
  3. For each combination, solve for $e$ and compute its posterior weight.
  4. Choose the solution with minimum weight (maximum probability).

- *Cost*: complexity grows as $2^k$. Usually small $k$ (0, 1, 2) already improves performance significantly.

== 3. Diagram: from BP to OSD

Below is a diagram showing how OSD uses BP output.

#figure(
  caption: [OSD workflow diagram],
  kind: "image",
  supplement: "Figure",
  block(
    height: 180pt,
    width: 100%,
    stroke: 0.5pt + gray.lighten(60%),
    radius: 5pt,
    inset: 10pt,
    align(center + horizon)[
      #box(width: 360pt, height: 160pt)[
        
        // 1. BP output
        #place(dx: 20pt, dy: 10pt, text(weight: "bold")[1. BP output LLR])
        // Draw bars for LLR magnitude
        #let bar(x, h, col, label) = {
          place(dx: x, dy: 60pt - h, rect(width: 15pt, height: h, fill: col))
          place(dx: x + 2pt, dy: 65pt, text(size: 8pt)[#label])
        }
        
        // Bits 1, 2, 3, 4, 5
        #bar(20pt, 40pt, blue.lighten(30%), "d1") // high confidence
        #bar(40pt, 10pt, gray.lighten(50%), "d2") // low confidence
        #bar(60pt, 35pt, blue.lighten(30%), "d3")
        #bar(80pt, 5pt,  gray.lighten(50%), "d4") // very low
        #bar(100pt, 45pt, blue.lighten(30%), "d5")
        
        #place(dx: 20pt, dy: 80pt, text(size: 9pt, style: "italic")[Long bars mean "confidence"])

        // Arrow to the right
        #place(dx: 130pt, dy: 50pt, text(size: 15pt)[$arrow.r$])

        // 2. Ordering and basis selection
        #place(dx: 160pt, dy: 10pt, text(weight: "bold")[2. Ordering and basis selection])
        
        // Rearranged matrix
        #place(dx: 160pt, dy: 30pt, [
          #rect(width: 80pt, height: 60pt, fill: white, stroke: 1pt + black)[
            #grid(
              columns: (1fr, 1fr),
              align: center + horizon,
              rect(width: 100%, height: 100%, fill: green.lighten(80%), stroke: none)[
                #text(size: 8pt)[$H_("reliable")$ \ (basis)]
              ],
              rect(width: 100%, height: 100%, fill: red.lighten(80%), stroke: none)[
                #text(size: 8pt)[$H_("others")$ \ (rest)]
              ]
            )
          ]
        ])
        #place(dx: 160pt, dy: 95pt, text(size: 8pt)[d5, d1, d3 ... d2, d4])
        
        // Arrow to the right
        #place(dx: 250pt, dy: 50pt, text(size: 15pt)[$arrow.r$])

        // 3. Linear solve
        #place(dx: 280pt, dy: 10pt, text(weight: "bold")[3. Linear solve])
        
        #place(dx: 280pt, dy: 40pt, block(width: 80pt)[
          #text(size: 10pt)[
            Solve: \
            $H_"basis" dot e_"basis" = s$
          ]
        ])
        
        #place(dx: 280pt, dy: 85pt, 
          rect(fill: luma(240), inset: 5pt, radius: 3pt, stroke: 0.5pt + black)[
            #text(size: 9pt, weight: "bold")[Get valid solution e]
          ]
        )
      ]
    ]
  )
)

== 4. Summary: BP+OSD

In QLDPC research, **BP+OSD** has become the de facto standard decoder configuration.

1.  **BP** handles sparse graph structure, using probabilistic info to converge near the true error.
2.  **OSD** uses linear algebra to enforce valid solutions and resolve degeneracy.

This combination keeps BP's low complexity ($O(n)$ to $O(n log n)$) while greatly improving the logical error threshold, approaching maximum-likelihood decoding (MLD).

#include"bp_note_trans.typ"
