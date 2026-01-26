#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/physica:0.9.5": *

#set cite(style: "apa")

// Time counter macro
#let globalvars = state("t", 0)
#let timecounter(minutes) = [
  #globalvars.update(t => t + minutes)
  #place(dx: 100%, dy: 0%, align(right, text(16pt, red)[#context globalvars.get()min]))
]

// Code block styling
#show raw.where(block: true): it => {
  par(justify: false, block(fill: rgb("#f0f0fe"), inset: 1.5em, width: 99%, text(it, 14pt)))
}

// Code box helper
#let codebox(txt, width: auto) = {
  box(inset: 10pt, stroke: blue.lighten(70%), radius: 4pt, fill: blue.transparentize(90%), text(14pt, txt), width: width)
}

// Global information configuration
#show: hkustgz-theme.with(
  config-info(
    title: [Getting Started with BPDecoderPlus],
    subtitle: [BP+OSD Decoder for Quantum Error Correction],
    author: [Si-Yuan Chen$""^*$, Meng-Kun Liu$""^*$, Shen Yang$""^*$ and Jin-Guo Liu$""^*$],
    date: datetime.today(),
  ),
)

#title-slide()
#outline-slide()

// ============================================================================
// Section 1: The Decoding Problem
// ============================================================================
= The Decoding Problem

== What Are We Decoding?
 
#v(20pt)
Quantum error correction protects logical qubits by encoding them in many physical qubits.
#align(center)[
  #box(stroke: blue, inset: 15pt, radius: 5pt)[
    Physical errors $arrow.r$ Syndrome measurements $arrow.r$ *Decoder* $arrow.r$ Logical error prediction
  ]
]

#v(20pt)

*The decoder's job:* Given syndrome measurements, infer what errors occurred and whether they flipped the logical qubit.

#v(20pt)

#grid(columns: 2, gutter: 30pt,
  [
    === Key Challenge
    - Degeneracy: identical syndromes.
    - Must predict the *logical effect*, not the exact physical errors
  ],
  [
    === Goal
    Minimize the *logical error rate* - the probability of incorrect logical predictions
  ]
)

== Circuit-Level vs Code-Capacity Noise
 

#align(center)[
  #table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Model*], [*Description*], [*Realism*]),
    [Code-capacity], [Errors only on data qubits, perfect measurements], [Simplified],
    [*Circuit-level*], [Errors on all operations, noisy measurements], [*Realistic*],
  )
]

#v(20pt)

BPDecoderPlus uses *circuit-level noise* - the realistic model where measurement operations themselves can fail.

#v(20pt)

#grid(columns: 2, gutter: 20pt,
  [
    === Code-Capacity
    - Single round of perfect measurements
    - Error model: $p$ on each data qubit
    - Useful for theoretical analysis
  ],
  [
    === Circuit-Level
    - Multiple rounds of noisy measurements
    - Errors on gates, idles, measurements
    - Required for real hardware
  ]
)

== Detection Events
 

We don't directly use raw syndrome bits. Instead, we use *detection events*:

#align(center)[
  #box(stroke: blue.lighten(50%), inset: 15pt, radius: 5pt, fill: blue.lighten(90%))[
    Detection event $=$ syndrome[round $t$] $xor$ syndrome[round $t-1$]
  ]
]

#v(15pt)

A detection event fires (value = 1) when the syndrome *changes* between rounds.

#v(15pt)

#grid(columns: 2, gutter: 30pt,
  [
    === Why Detection Events?
    - Raw syndromes are noisy (measurement errors)
    - Detection events *cancel* measurement errors that persist across rounds
    - Stim's detector error model (DEM) is defined in terms of detection events
  ],
  [
    === Example Timeline
    #align(center)[
      #table(
        columns: (auto, auto, auto, auto),
        inset: 8pt,
        align: center,
        table.header([*Round*], [$t-1$], [$t$], [$t+1$]),
        [Syndrome $s$], [0], [1], [1],
        [Detection $d$], [—], [#text(red)[1]], [#text(green)[0]],
      )
    ]
    #v(5pt)
    #text(10pt)[Detection $d_t = s_t xor s_(t-1)$]
  ]
)

// ============================================================================
// Section 2: Pipeline Overview
// ============================================================================
= Pipeline Overview

== Pipeline Steps
 

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: horizon,
    table.header([*Step*], [*Input*], [*Output*], [*Purpose*]),
    [1. Generate Circuit], [Parameters $(d, r, p)$], [`.stim` file], [Define noisy quantum operations],
    [2. Extract DEM], [`.stim` circuit], [`.dem` file], [Map errors $arrow.r$ detections],
    [3. Build $H$ Matrix], [`.dem` file], [$H$, priors, obs_flip], [Decoder input format],
    [4. Sample Syndromes], [`.stim` circuit], [`.npz` file], [Training/test data],
    [5. Decode], [$H$ + syndromes], [Predictions], [Infer logical errors],
  )
]

== The DEM: Detector Error Model
 
#v(20pt)
The DEM is the crucial link between physical errors and observable detection events.

#v(10pt)

#align(center)[
  #box(stroke: green, inset: 15pt, radius: 5pt, fill: green.lighten(90%))[
    error(0.01) D0 D5 L0
  ]
]


This entry means: _"There's a 1% probability of an error that triggers detectors 0 and 5, and flips the logical observable."_


#grid(columns: 2, gutter: 30pt,
  [
    === DEM Specifies
    - *What errors* can occur (each line is one error mechanism)
    - *Which detectors* fire (D0, D1, etc.)
    - *Logical effect* 
    - *Probability* (the number in parentheses)
  ],
  [
    === Generated By
    - Stim's circuit analysis @Gidney2021
    - Automatic error propagation
    - Decomposition of correlated errors
  ]
)

// ============================================================================
// Section 3: BP and Why It Fails
// ============================================================================
= Belief Propagation & Why It Fails

== Factor Graph from $H$ Matrix
 

The parity check matrix $H$ defines a *factor graph* (Tanner graph) @Kschischang2001:

#grid(columns: 2, gutter: 20pt,
  [
    #v(10pt)
    - *Variable nodes* (circles): Error mechanisms (columns of $H$)
    - *Check nodes* (squares): Detectors (rows of $H$)
    - *Edges*: $H[i,j] = 1$ connects detector $i$ to error $j$
    
    #v(15pt)
    
    *Example:* $H = mat(1, 1, 0, 1; 0, 1, 1, 1)$
    
    - 4 error variables, 2 detectors
    - $e_2$ and $e_4$ connected to both checks
  ],
  [
    #align(center)[
      #table(
        columns: (auto, auto, auto, auto, auto),
        inset: 8pt,
        align: center,
        stroke: 0.5pt,
        table.header([], [$e_1$], [$e_2$], [$e_3$], [$e_4$]),
        [$D_0$], [1], [1], [0], [1],
        [$D_1$], [0], [1], [1], [1],
      )
    ]
    #v(10pt)
    #align(center, text(10pt)[
      _$D_0$ checks: $e_1 xor e_2 xor e_4$_\
      _$D_1$ checks: $e_2 xor e_3 xor e_4$_
    ])
  ]
)

== Message Passing Intuition
 

BP iteratively passes "beliefs" between nodes @Pearl1988:

#v(10pt)

#align(center)[
  #box(stroke: gray.lighten(50%), inset: 15pt, radius: 5pt)[
    #text(blue)[$e$] #h(10pt) $limits(arrow.r.long)^(mu_(e arrow D))$ #h(10pt) #text(orange)[$D$] #h(30pt) #text(orange)[$D$] #h(10pt) $limits(arrow.r.long)^(mu_(D arrow e))$ #h(10pt) #text(blue)[$e$]
  ]
]

#v(10pt)

#grid(columns: 3, gutter: 15pt,
  [
    === Step 1
    *Variable $arrow$ Check*
    
    "Here's my current probability of being an error"
  ],
  [
    === Step 2
    *Check $arrow$ Variable*
    
    "Given what others told me, here's what you should be to satisfy the parity"
  ],
  [
    === Step 3
    *Repeat*
    
    Until convergence or max iterations reached
  ]
)

#v(10pt)

After convergence, each variable has a *marginal probability* of being an error.

== The Degeneracy Problem
 

#align(center)[
  #box(stroke: red, inset: 15pt, radius: 5pt, fill: red.lighten(90%))[
    *BP works perfectly on trees, but quantum codes have loops!*
  ]
]

#v(15pt)

On loopy graphs, BP can:
- Fail to converge
- Converge to wrong probabilities
- Output invalid solutions ($H dot e eq.not$ syndrome)

#v(15pt)

#grid(columns: 2, gutter: 20pt,
  [
    === Most Critically
    BP outputs *probabilities*, but rounding them doesn't guarantee a valid solution.
    
    #v(10pt)
    
    === Why Quantum Codes Are Hard
    - Classical LDPC: Each error has unique syndrome signature
    - *Quantum surface codes*: Multiple error patterns produce the *same syndrome* (degeneracy)
    - BP gets "confused" by equivalent solutions
  ],
  [
    #align(center)[
      #image("images/bp_failure_demo.png", width: 100%)
    ]
    #align(center, text(10pt)[_BP's marginals produce invalid error pattern_])
  ]
)

// ============================================================================
// Section 4: OSD Post-Processing
// ============================================================================
= OSD Post-Processing

== The Key Insight
 

OSD (Ordered Statistics Decoding) @Fossorier1995 forces a *unique, valid solution* by treating decoding as a system of linear equations:

#align(center)[
  #box(stroke: blue, inset: 20pt, radius: 5pt, fill: blue.lighten(90%))[
    $H dot e = s quad (mod 2)$
  ]
]

#v(15pt)

Given syndrome $s$, find error vector $e$ that satisfies this constraint.

#v(15pt)

#grid(columns: 2, gutter: 30pt,
  [
    === BP Provides
    - Soft information (probabilities)
    - *Unreliable* hard decisions
    - No validity guarantee
  ],
  [
    === OSD Guarantees
    - *Valid* solutions ($H dot e = s$)
    - Uses BP's confidence to guide search
    - Polynomial time complexity
  ]
)

== OSD-0 Algorithm in 3 Steps
 

#align(center)[
  #table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: (left, left, left),
    stroke: 0.5pt,
    table.header([*Step*], [*Operation*], [*Result*]),
    [1. Sort], [Order columns by $|"LLR"|$ descending], [High confidence columns first],
    [2. Row Reduce], [Gaussian elimination on $H$], [$[I | P]$ form with pivots],
    [3. Solve], [Back-substitution with $s$], [Valid error vector $e$],
  )
]

#v(10pt)

#grid(columns: 2, gutter: 20pt,
  [
    ```python
    # Step 1: Sort
    order = argsort(|LLR|, descending)
    H_sorted = H[:, order]
    ```
  ],
  [
    ```python
    # Steps 2-3: Reduce and solve
    H_reduced, pivots = row_reduce(H_sorted)
    e = back_substitute(H_reduced, s)
    ```
  ]
)

*Result:* A valid codeword that respects BP's confident decisions.

== OSD-W: Search for Better Solutions
 

OSD-0 fixes non-pivot bits to BP's decision. *OSD-W* searches over $2^W$ combinations of the $W$ least confident non-pivot bits.

#v(15pt)

#align(center)[
  #box(stroke: blue.lighten(50%), inset: 12pt, radius: 5pt)[
    BP Marginals $arrow.r$ Sort by Confidence $arrow.r$ Gaussian Elimination $arrow.r$ #box(stroke: purple, inset: 5pt, radius: 3pt)[OSD-0 / OSD-W] $arrow.r$ Best Solution
  ]
]

#v(10pt)

#align(center)[
  #table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, center, left),
    stroke: 0.5pt,
    table.header([*Method*], [*Candidates*], [*Trade-off*]),
    [OSD-0], [$1$], [Fast, single solution],
    [OSD-$W$], [$2^W$], [Search $W$ least confident bits],
    [OSD-CS], [$1 + k + binom(W, 2)$], [Efficient: weight-0,1,2 patterns],
  )
]

#v(5pt)

#grid(columns: 2, gutter: 30pt,
  [
    === OSD Order Trade-off
    - *OSD-0*: Fast, single solution
    - *OSD-10*: $2^{10}$ candidates, better accuracy
    - *OSD-20*: Near-optimal, slower
  ],
  [
    === Selection Criterion
    Pick the candidate with lowest *soft-weighted cost*:
    $ "cost"(e) = sum_i |"LLR"_i| dot e_i $
  ]
)

== The Fix in Action
 

#align(center)[
  #image("images/osd_success_demo.png", width: 85%)
]

#align(center)[
  _The same syndrome, now decoded correctly with OSD post-processing._
  
  *OSD guarantees* $H dot e = s$
]

// ============================================================================
// Section 5: Demo and Summary
// ============================================================================
= Demo & Summary


== Threshold Results
 
 #v(20pt)
The *threshold* is the physical error rate below which larger codes perform better.

#grid(columns: 2, gutter: 20pt,
  [
    #align(center)[
      #image("images/threshold_plot.png", width: 100%)
    ]
  ],
  [

    #align(center)[
      #table(
        columns: (auto, auto, auto),
        inset: 8pt,
        align: horizon,
        table.header([*Decoder*], [*Threshold*], [*Notes*]),
        [BP (damped)], [N/A], [Fast, limited by loops],
        [*BP+OSD*], [$tilde 0.7%$], [Near-optimal],
        [MWPM], [$tilde 0.7%$], [Gold standard @Higgott2023],
      )
    ]

    BP+OSD achieves *near-optimal threshold* with good computational efficiency.
  ]
)

== Tropical Results
 
 #v(20pt)
The *threshold* for tropical tensor network decoder require more memory. Further approximate contraction method?

#grid(columns: 2, gutter: 20pt,
  [
    #align(center)[
      #image("images/tropical_threshold_plot.png", width: 100%)
    ]
  ],
  [

    #align(center)[
      #table(
        columns: (auto, auto, auto),
        inset: 8pt,
        align: horizon,
        table.header([*Decoder*], [*Threshold*], [*Notes*]),
        [BP (damped)], [N/A], [Fast, limited by loops],
        [*BP+OSD*], [$tilde 0.7%$], [Near-optimal],
        [MWPM], [$tilde 0.7%$], [Gold standard @Higgott2023],
      )
    ]

    Tropical is too resource consuming!
  ]
)

== Summary
 

#align(center)[
  #box(stroke: blue, inset: 20pt, radius: 5pt)[
    *BPDecoderPlus: Fast and accurate decoding for quantum error correction*
  ]
]

#v(15pt)

#grid(columns: 2, gutter: 30pt,
  [
    === Key Takeaways
    1. *Decoding* = syndrome $arrow$ error prediction
    2. *DEM* maps physical errors to detections
    3. *BP* provides soft information but may fail
    4. *OSD* guarantees valid solutions
    5. *BP+OSD* achieves near-optimal threshold
  ],
  [
    === Next Steps
    - Try: `uv run python examples/minimal_example.py`
    - CLI: `uv run bpdecoder --help`
    - Docs: `docs/usage_guide.md`
    - Math: `docs/mathematical_description.md`
  ]
)

#v(20pt)

#align(center)[
  GitHub: #link("https://github.com/GiggleLiu/BPDecoderPlus")[`GiggleLiu/BPDecoderPlus`]
]

==

#bibliography("refs.bib")
