#import "@preview/peace-of-posters:0.5.6" as pop
#import "@preview/cetz:0.4.1": canvas, draw
#import "@preview/qec-thrust:0.1.1": *

#show link: set text(blue)
#set page("a0", margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)
#set text(size: pop.layout-a0.at("body-size"))
#let box-spacing = 1.2em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

#pop.title-box(
  "BPDecoderPlus: Circuit-Level Quantum Error Correction with Belief Propagation and Tropical Tensor Networks",
  authors: [Si-Yuan Chen$""^*$, Meng-Kun Liu$""^*$, Shen Yang$""^*$ and Jin-Guo Liu$""^*$],
  institutes: text(36pt)[
  ],
  title-size: 1.3em,
)

#columns(2,[

  #pop.column-box(heading: "Abstract")[
Quantum error correction (QEC) is essential for fault-tolerant quantum computing. We present *BPDecoderPlus*, a Python package implementing two complementary approaches for decoding surface codes under circuit-level noise:

1. *BP+OSD Decoder*: Belief propagation with ordered statistics decoding post-processing, achieving near-optimal performance on quantum LDPC codes.

2. *Tropical Tensor Networks*: Exact Most Probable Explanation (MPE) computation via tropical semiring contraction, providing optimal solutions for moderate-size instances.

Our implementation correctly resolves the circuit-level error threshold at $approx 0.7%$ for rotated surface codes, validating against established literature. The package features GPU acceleration via PyTorch, comprehensive CLI tools, and integration with Google's Stim simulator.
  ]

  #pop.column-box(heading: "Rotated Surface Code")[
#grid(columns: 2, gutter: 20pt,
canvas(length: 2cm, {
  import draw: *
  surface-code((0, 0), size: 2.5, 3, 3, name: "sc")
}),
box[
  *Detection events* (not raw syndromes):
  - Measurement errors flip syndrome values
  - Event = XOR of consecutive measurements
  - Localizes errors in space-time

  *Detector Error Model* (DEM):
  - Error probabilities per fault
  - Detector-error associations
  - Observable flip annotations
]
)

#align(center)[
  #box(stroke: 2pt, inset: 12pt, radius: 5pt)[
    Detection Events $arrow.r$ *Decoder* $arrow.r$ Observable Flip Prediction
  ]
]
  ]

  #let hba = pop.uni-fr.heading-box-args
  #hba.insert("stroke", (paint: gradient.linear(blue, purple, red), thickness: 10pt))
  #let bba = pop.uni-fr.body-box-args
  #bba.insert("inset", 30pt)
  #bba.insert("stroke", (paint: gradient.linear(blue, purple, red), thickness: 10pt))

  #pop.column-box(heading: "BP+OSD Decoder", stretch-to-next: true)[
*Belief Propagation* iteratively passes messages on a factor graph to compute marginal probabilities. For QEC, the factor graph is derived from the parity check matrix $H$.

#grid(columns: 2, gutter: 40pt,
canvas(length: 1.2cm, {
  import draw: *
  // Variable nodes (circles)
  for (i, x) in ((0, -2), (1, 0), (2, 2)) {
    circle((x, 2), radius: 0.4, name: "v" + str(i), stroke: 2pt, fill: rgb("#e3f2fd"))
    content((x, 2), [$e_#i$])
  }
  // Factor nodes (squares)
  for (i, x) in ((0, -1), (1, 1)) {
    rect((x - 0.4, -0.4), (x + 0.4, 0.4), name: "f" + str(i), stroke: 2pt, fill: rgb("#fff3e0"))
    content((x, 0), [$d_#i$])
  }
  // Edges
  line("v0", "f0", stroke: 2pt)
  line("v1", "f0", stroke: 2pt)
  line("v1", "f1", stroke: 2pt)
  line("v2", "f1", stroke: 2pt)
  content((0, -1.5), [Factor Graph])
}),
box[
  *Message passing:*
  $ mu_(v arrow f) = product_(f' in N(v) \\ f) mu_(f' arrow v), mu_(f arrow v) = sum_(bold(x): x_v = 0,1) psi_f (bold(x)) product_(v' in N(f) \\ v) mu_(v' arrow f) $
]
)

  #figure(
    image("../images/bp_failure_demo.png", width: 100%),
    caption: [BP alone fails due to degeneracy: the decoder outputs an invalid solution that does not satisfy the syndrome.]
  )
  *Ordered Statistics Decoding (OSD)* post-processes BP output:
1. Sort variables by BP reliability
2. Fix most reliable bits using Gaussian elimination
3. Exhaustively search remaining bits (OSD-$w$ searches $w$ bits)

#figure(
  canvas(length: 2cm, {
    import draw: *

    // Original matrix
    rect((-4, -1), (-1, 1), fill: rgb("#f0f0f0"), name: "H")
    content("H", $H$)
    content((-2.5, -1.5), text(size: 30pt)[$m times n$])

    // Arrow
    line((-0.5, 0), (0.5, 0), mark: (end: ">"))
    content((0, 0.5), text(size: 30pt)[split])

    // Basis submatrix
    rect((1, -1), (3, 1), fill: rgb("#e0ffe0"), name: "HS")
    content("HS", $H_([S])$)
    content((1.75, -1.5), text(size: 30pt)[$m times r$])
    content((1.75, -2), text(size: 30pt)[invertible!])

    // Remainder submatrix
    rect((3.5, -1), (5, 1), fill: rgb("#ffe0e0"), name: "HT")
    content("HT", $H_([T])$)
    content((4, -1.5), text(size: 30pt)[$m times k'$])
  }),
  caption: [Splitting $H$ into basis and remainder parts]
)
  ]


#colbreak()

  #pop.column-box(heading: "BP+OSD Threshold Results")[
#align(center)[
  #image("threshold_comparison.png", width: 95%)
]

#table(
  columns: 4,
  align: center,
  stroke: 0.5pt,
  table.header([*Noise Model*], [*BP Only*], [*BP+OSD*], [*Optimal*]),
  [Code capacity], [N/A], [$approx 9.9%$], [$10.3%$],
  [Circuit-level], [N/A], [$approx 0.7%$], [$approx 1%$],
)

#grid(columns: 2, gutter: 20pt,
box[
  *Configuration:*
  - BP: 60 iterations, min-sum
  - Damping: 0.2, OSD order: 10
],
box[
  *Validation:*
  - Matches ldpc library @Higgott2023
  - Curves cross at threshold
]
)
  ]

  #pop.column-box(heading: "Tropical Tensor Networks for MPE")[
The *Most Probable Explanation* (MPE) problem finds the most likely error pattern given observations. We solve this exactly using tropical tensor networks.

*Tropical Semiring:* $(RR union {-infinity}, max, +)$
- Addition $arrow.r$ max operation
- Multiplication $arrow.r$ standard addition

#grid(columns: 2, gutter: 30pt,
box[
  *Standard tensor contraction:*
  $ C_(i k) = sum_j A_(i j) dot B_(j k) $
],
box[
  *Tropical contraction:*
  $ C_(i k) = max_j (A_(i j) + B_(j k)) $
]
)

For probabilistic graphical models in log-space:
$ log P(bold(x)) = sum_f log psi_f (bold(x)_f) $

#highlight[Tropical contraction computes $max_(bold(x)) log P(bold(x))$ exactly!]

*Implementation highlights:*
- Uses `omeco` for optimal contraction ordering
- PyTorch backend with GPU support
- Backtracking recovers the optimal assignment
- Complexity: $O(2^(text("treewidth")))$
  ]

  #pop.column-box(heading: "Software Architecture")[
#align(center)[
#box(stroke: 1pt, inset: 10pt, radius: 3pt)[
  Stim Circuit $arrow.r$ DEM $arrow.r$ Factor Graph $arrow.r$ BP/Tropical $arrow.r$ Prediction
]
]

*Key features:*
- *Stim integration*: Generate noisy circuits for rotated surface codes
- *DEM parsing*: Two-stage processing (separator splitting + hyperedge merging)
- *PyTorch backend*: GPU-accelerated batch inference
- *CLI tools*: `generate-noisy-circuits` for dataset creation
- *Modular design*: BP and Tropical modules are independent

#highlight[Developed with *vibe coding*: human-AI collaboration using Claude Code accelerated development from concept to working threshold plots.]

#link("https://github.com/TensorBFS/BPDecoderPlus")[github.com/TensorBFS/BPDecoderPlus]
  ]

  #pop.column-box(heading: "References", stretch-to-next: true)[
    #bibliography("bibliography.bib", title: none)
  ]
])

#pop.bottom-box()[
  #align(right, [#align(horizon, grid(columns: 5, column-gutter: 30pt, image("github-dark.png", width: 70pt), "TensorBFS/BPDecoderPlus", h(50pt), image("email.png", width: 70pt), "jinguoliu@hkust-gz.edu.cn"))])
]
