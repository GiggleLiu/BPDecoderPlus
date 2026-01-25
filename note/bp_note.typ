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
= Note

== 1. Storing quantum states on a classical computer <quantum-storage> 

Classical bits are definite like `10101`, but a quantum state contains superposition and entanglement. There are two main ways to store it: a "general brute-force" method and a "QEC-optimized efficient" method.

=== 1.1 General method: state vector (State Vector) -- exponential blowup

This is the most intuitive but extremely expensive approach.
For $n$ qubits, the system can be in a superposition of all $2^n$ basis states:

$
  |psi > = alpha_0 |00...0 > + alpha_1 |00...1 > + dots + alpha_(2^n - 1) |11...1 >
$

In classical memory, we must store **every** $alpha_i$.
- Each $alpha_i$ is complex (usually 2 64-bit floats, i.e. 16 bytes).
- *Required memory*: $16 times 2^n$ bytes.

#table(
  columns: (1fr, 1fr, 2fr),
  inset: 8pt,
  align: horizon,
  stroke: none,
  table.header([*Number of qubits ($n$)*], [*Number of complex values*], [*Required memory*]),
  table.hline(stroke: 0.5pt),
  [10], [1,024], [16 KB (negligible)],
  [30], [~1 billion], [16 GB (high-end PC limit)],
  [50], [~1,125 trillion], [16 PB (supercomputer limit)],
  [100], [$1.2 times 10^30$], [More than the number of atoms on Earth]
)

#note("Conclusion")[
  This method can only simulate small circuits below 40-50 qubits and cannot simulate surface codes with thousands of qubits.
]

=== 1.2 QEC-specific method: stabilizer tableau (Stabilizer Tableau) -- efficient storage

In quantum error correction research (especially with Pauli errors and Clifford gates), we use the **Gottesman-Knill theorem**.

If the circuit contains only Clifford gates (H, CNOT, S, Pauli), we **do not need** to store wavefunction amplitudes; we only need to track **the evolution of stabilizer operators**.

=== 1.3 Data structure: binary matrix (Tableau)
We only need to store an $n times 2n$ **binary matrix** (0 and 1).
- No complex numbers.
- Each row is a stabilizer generator.
- Each column indicates whether the stabilizer acts as $X$ or $Z$ on qubit $i$.

Assume there are $n$ qubits. We use two binary bits $(x_{i}, z_{i})$ to represent the Pauli operator on qubit $i$:
- $I arrow (0, 0)$
- $X arrow (1, 0)$
- $Z arrow (0, 1)$
- $Y arrow (1, 1)$ (because $Y = i X Z$)

#note("Binary representation of Pauli operators")[
In the Pauli frame, a single-qubit Pauli operator can be written as  
$X^(x_i) Z^(z_i)$, where $x_i, z_i ∈ {0, 1}$.

The mapping is:
- $I → (0, 0)$
- $X → (1, 0)$
- $Z → (0, 1)$
- $Y → (1, 1)$ (because $Y = i · X · Z$, ignoring global phase)

Therefore, a stabilizer operator on $n$ qubits  
$S = P_1 ? P_2 ? ? ? P_n$ 
can be represented by a length-$2n$ binary vector:
$(x_1, …, x_n ∥ z_1, …, z_n)$.
]
#note("Meaning of tableau dimensions")[

- **Rows (about $n$)**: number of stabilizer generators.
- **Columns ($2n$)**: for each stabilizer,  
  - $n$ $X$ components,  
  - $n$ $Z$ components.

So the tableau is an $n × 2n$ binary matrix overall.]

=== 1.4 Example
Let $n=2$, with stabilizers $X_1 X_2$ and $Z_1 Z_2$.
The stored tableau looks like:

$
  mat(
    delim: "[",
    "x1", "x2", "|", "z1", "z2", "r (phase)";
    1, 1, "|", 0, 0, 0;
    0, 0, "|", 1, 1, 0
  )
$
- First row $(1,1,0,0) arrow X_1 X_2$
- Second row $(0,0,1,1) arrow Z_1 Z_2$

=== 1.5 Advantages
- *Memory complexity*: $O(n^2)$. Even for $n=10,000$, only a few MB of binary matrix storage.
- *Compute speed*: bitwise operations (XOR) are extremely fast.

=== 1.6 Summary: which storage is used in BP decoding?

In LDPC and surface-code decoding (BP+OSD), we typically assume a Pauli channel error model (only X, Y, Z errors, no small rotations).

Therefore simulators (e.g. Python's `stim` library) use *Method 2 (stabilizer tableau)*.

1. **Storage**: do not store $|psi >$, only the current stabilizer generator matrix.
2. **Errors**: just flip binary bits in the tableau (0 to 1).
3. **Syndrome**: computed by matrix multiplication $H dot e^T$, giving a classical binary string `10101...`.
4. **BP input**: BP receives that classical `10101` syndrome and infers the underlying errors.

#link(<quantum-storage-back>)[#text(fill:blue)[*BACK?*]]

== 2. Why must stabilizers commute? (Commute)<Commute>
  This is the physical foundation for QEC to work, for two main reasons:

=== 2.1 Possibility of simultaneous measurement
In quantum mechanics, the Heisenberg uncertainty principle tells us:
*Only when two operators $A$ and $B$ commute (i.e. $[A, B] = "AB" - "BA" = 0$) can we simultaneously determine their measurement outcomes.*

- **If they do not commute**: measuring $S_1$ disturbs $S_2$.
  For example, if we first measure $S_1$ and get $+1$, then measure $S_2$, $S_1$ might flip back to $-1$ (or the system collapses to a state that is not an eigenstate of $S_1$).
- **In QEC**: we need to extract all syndromes at once to diagnose errors. If stabilizers "fight", we cannot obtain a stable, consistent syndrome. The measurement itself would create new errors.

=== 2.2 Existence of common eigenstates
The logical space (code space) is defined as the $+1$ common eigenspace of all stabilizers:
$
  S_i |psi_L > = +1 |psi_L >, quad forall i
$
Linear algebra tells us that a set of operators has a common eigenbasis only if they mutually commute.
If they do not commute, there is no state that satisfies all $S_i$ constraints, and the code space cannot be defined.

=== 2.3 Mathematical derivation: why does $H_X dot H_Z^T = 0$ imply commutation?

This comes from Pauli algebra:
- On the same qubit: $X Z = - Z X$ (anticommute, sign -1).
- On different qubits: $X_i Z_j = Z_j X_i$ (commute).

Suppose an X stabilizer $S_x$ and a Z stabilizer $S_z$ act on $n$ qubits. Whether they commute depends on how many positions they "collide" (both non-identity).
$
  S_x S_z = (-1)^k S_z S_x
$
where $k$ is the number of qubits where $S_x$ applies $X$ and $S_z$ applies $Z$ (overlap count).

- For $S_x S_z = S_z S_x$, we need $(-1)^k = 1$, i.e. **$k$ must be even**.
- In binary matrix multiplication, the dot product of a row of $H_X$ with a column of $H_Z^T$ computes exactly this overlap count $k$ (mod 2).

Therefore, requiring $H_X dot H_Z^T = 0 mod 2$ means every X-check and Z-check overlap an even number of times, guaranteeing physical commutation.

#note("Commutation check")[

Below is a minimal CSS example to verify: when $H_X · H_Z^T = 0$ (mod 2), the corresponding X-stabilizer and Z-stabilizer do commute.

Example setup ($n = 4$)

Take one X-check row vector and one Z-check row vector:

$h_x = [1, 1, 0, 0]$  
$h_z = [1, 1, 1, 1]$

Their Pauli stabilizers are:

$S_x = X_1 X_2$  
$S_z = Z_1 Z_2 Z_3 Z_4$

Here "1" means the Pauli acts at that position ($X$ for $S_x$, $Z$ for $S_z$),  
"0" means identity $I$.

Dot product: $h_x · h_z^T$

Dot product (mod 2):

$h_x · h_z^T
= (1·1 + 1·1 + 0·1 + 0·1) mod 2
= (1 + 1 + 0 + 0) mod 2
= 2 mod 2
= 0$

So this pair satisfies $H_X · H_Z^T = 0$ (for this row/column pair).

Count of "collisions" $k$

A "collision" means $X$ (from $S_x$) and $Z$ (from $S_z$) act on the same qubit.  
In this example:

- Qubit 1: $X_1$ and $Z_1$ collide (1 time)
- Qubit 2: $X_2$ and $Z_2$ collide (1 more time)
- Qubits 3,4: $S_x$ is $I$, no collision

So the collision count is $k = 2$ (even).

Verify commutation with Pauli algebra

Using $X Z = - Z X$ on the same qubit and commutation on different qubits:

$S_x S_z
= (X_1 X_2)(Z_1 Z_2 Z_3 Z_4)$

When swapping to group same-qubit terms, each swap $X_i$ with $Z_i$ introduces a $-1$:

- Qubit 1: $X_1 Z_1 = - Z_1 X_1$ contributes $-1$
- Qubit 2: $X_2 Z_2 = - Z_2 X_2$ contributes another $-1$

Thus the total sign is $(-1)^k = (-1)^2 = +1$, so:

$S_x S_z = (+1) S_z S_x = S_z S_x$

So they **commute**.
]

#link(<Commute-back>)[#text(fill:blue)[*BACK?*]] 

== 3. Scientific basis and sources of the prior (Prior) <Prior>

*"Before we even compute, why are we allowed to assume every bit has a fixed error probability $p$? Is that scientific?"* In Bayesian statistics, this is called a **prior**. In quantum error correction, introducing it is not only scientific, but necessary, for the following reasons:

=== 3.1 Physical source: obtained through hardware calibration
This $p$ is not a number we make up during decoding; it comes from **measured experimental data**.
- Before a quantum chip is used, experimentalists run benchmarks (e.g., *Randomized Benchmarking* or *Gate Set Tomography*).
- These tests tell us the average gate fidelity (e.g., 99.9%).
- So we obtain a physical error rate $p approx 0.001$.
- **Scientific basis**: it represents our **statistical knowledge** of hardware quality.

=== 3.2 Mathematical necessity of Bayesian inference
BP is essentially Bayesian inference. By Bayes' theorem:
$
  P("error" | "phenomenon") ∝ P("phenomenon" | "error") dot P("error")
$
- *"Phenomenon"* is the observed syndrome (which stabilizers fired).
- *"Error"* is the error vector $e$ we want to infer.
- *$P("error")$* is the prior probability $p$.

If we do not include $p$, we are effectively assuming "error" and "no error" are equally likely (i.e., $p=0.5$), which makes $L=0$. That means there is no initial bias, and the decoder will struggle to converge because it has no baseline to judge whether a strange syndrome comes from a rare complex error or a common simple one.

=== 3.3 Why does the LLR formula look like that?
The formula $L = ln((1-p)/p)$ is a **weighting system**.
- If $p$ is small (good hardware, e.g. $10^(-3)$), $L approx ln(1000) approx 6.9$, so the initial confidence is high.
- If $p$ is large (poor hardware, e.g. $10^(-1)$), $L approx ln(9) approx 2.2$, so the initial confidence is lower.

This tells BP: *"Unless nearby checks (evidence) strongly accuse this bit, since $p$ is small you should tend to believe it is innocent."*

=== 3.4 Limitations
Of course, a simple $p$ model is not always realistic: it usually assumes errors are **i.i.d.** (independent and identically distributed).
But in real hardware, errors are often correlated (e.g., cosmic rays can flip a whole region, or a 2-qubit gate can cause two bits to fail together). Advanced decoders use more complex **correlated noise models** to initialize LLRs, rather than a single $p$.

#link(<Prior-back>)[#text(fill:blue)[*BACK?*]] 
