# STRAW Project Report (Presentation Draft)

## 1) Project Name
**STRAW**: **S**ample-**T**uned **R**ank-**A**ugmentative **W**eights

STRAW is a dynamic adaptation method for LLMs where LoRA weights are generated per input sample instead of using one fixed adapter for all prompts.

---

## 2) What We Built

We built a CNN-based hypernetwork pipeline on top of `mistralai/Mistral-7B-Instruct-v0.3` that:
- reads prefix hidden states from the residual stream,
- generates per-layer low-rank factors `A_l(x)` and `B_l(x)`,
- injects dynamic LoRA updates into transformer attention projections (primarily `v_proj`).

This makes the adapter **input-conditioned** and **domain-adaptive**.

---

## 3) Core Idea and Equation

For each transformer layer `l`, STRAW constructs:

\[
\Delta W_l(x) = \frac{\alpha}{r} B_l(x) A_l(x)
\]

and uses it in the projection path as:

\[
y = (W + \Delta W_l(x))h
\]

where:
- `x` = current sample/prompt,
- `h` = hidden state at that projection,
- `r` = low rank,
- `alpha` = LoRA scaling.

Unlike static LoRA, `\Delta W_l` changes per sample.

---

## 4) How It Works (Flow)

1. Input prompt enters the base transformer.
2. Prefix token hidden states are collected from the residual stream.
3. Prefix states are pooled and normalized.
4. CNN hypernetwork maps prefix features to per-layer latent features.
5. Heads produce low-rank factors `A_l(x), B_l(x)` for targeted layers.
6. Dynamic LoRA deltas are injected into attention projections.
7. Model generates outputs with sample-conditioned adaptation.

---

## 5) Architecture Diagram (LaTeX/TikZ)

Use this directly in slides/paper for a clean generalized figure:

```latex
\documentclass[tikz,border=7pt]{standalone}
\usepackage{amsmath,amssymb}
\usetikzlibrary{arrows.meta,positioning,fit,calc,backgrounds}

\begin{document}
\begin{tikzpicture}[
    >=Latex,
    font=\small,
    node distance=6mm and 10mm,
    box/.style={draw, rounded corners=2mm, thick, align=center, inner sep=4pt},
    io/.style={box, fill=blue!10, draw=blue!60!black, minimum width=35mm},
    hyp/.style={box, fill=orange!14, draw=orange!70!black, minimum width=42mm},
    tr/.style={box, fill=teal!10, draw=teal!60!black, minimum width=48mm},
    lora/.style={box, fill=purple!10, draw=purple!65!black, minimum width=52mm},
    eqb/.style={box, fill=gray!8, draw=black!45, text width=70mm, align=left},
    arr/.style={-Latex, thick},
    darr/.style={dotted, very thick}
]

\node[io] (tokens) {\textbf{Prefix tokens}\\instruction / prompt prefix};
\node[tr, below=of tokens] (prefixenc) {\textbf{Prefix hidden states}\\$H^{\text{pref}}\in\mathbb{R}^{T_p\times d}$};
\draw[arr] (tokens) -- (prefixenc);

\node[hyp, right=18mm of prefixenc] (pool) {\textbf{Pool + Normalize}\\$\tilde s=\mathrm{LN}(\mathrm{Pool}(H^{\text{pref}}))$};
\node[hyp, below=of pool] (hyper) {\textbf{Hypernetwork}\\$z_\ell = g_\ell(\tilde s)$};
\node[hyp, below=of hyper] (heads) {\textbf{LoRA factor heads}\\$A_\ell(x),\,B_\ell(x)$ for each layer $\ell$};
\draw[arr] (prefixenc.east) -- (pool.west);
\draw[arr] (pool) -- (hyper);
\draw[arr] (hyper) -- (heads);

\node[tr, right=20mm of pool] (layer) {\textbf{Transformer block $\ell$}\\Self-attention + MLP};
\node[lora, below=of layer] (inject) {\textbf{Dynamic adapter integration}\\Inject into target projection(s)\\(e.g., $v_{\text{proj}}$)};
\draw[arr] (layer) -- (inject);
\draw[arr] (heads.east) -- node[above,sloped,pos=.55]{\scriptsize sample-conditioned adapters} (inject.west);

\node[eqb, below=10mm of heads] (eq) {
\[
\Delta W_\ell(x)=\frac{\alpha}{r}B_\ell(x)A_\ell(x),\qquad
A_\ell\in\mathbb{R}^{r\times d},\;B_\ell\in\mathbb{R}^{d\times r}
\]
\[
y = (W + \Delta W_\ell(x))\,h
\]
\textbf{Interpretation:} adapter parameters are generated from prefix-conditioned features, so the effective weights vary across samples.
};
\draw[darr] (inject.south) -- ++(0,-5mm) -| (eq.east);

\begin{scope}[on background layer]
\node[draw=blue!55!black, rounded corners=2mm, line width=.8pt, inner sep=4mm,
      fit=(tokens)(prefixenc), label={[blue!60!black]above:\scriptsize Prefix encoding}] {};
\node[draw=orange!65!black, rounded corners=2mm, line width=.8pt, inner sep=4mm,
      fit=(pool)(hyper)(heads), label={[orange!70!black]above:\scriptsize Hypernetwork generation}] {};
\node[draw=teal!55!black, rounded corners=2mm, line width=.8pt, inner sep=4mm,
      fit=(layer)(inject), label={[teal!60!black]above:\scriptsize Transformer integration}] {};
\end{scope}

\node[draw=none, font=\bfseries\large, text=black!80] at ($(tokens.north)!0.5!(layer.north)+(0,1.1)$)
{Prefix-Conditioned Hypernetwork Integration};
\end{tikzpicture}
\end{document}
```

---

## 6) Training/Eval Setup Used in This Run

Run reference: `experiments/runs/main_gen_1k.sh` with `RUN_TAG=straw_1k_v2`

- Train samples: `1000` per dataset
- Eval/Test samples: `100`
- Epochs: `5`
- Default eval generation length: `256` tokens
- STRAW config highlights:
  - `model_type: cnn`
  - `straw_rank: 16`
  - `lora_alpha: 16`
  - `layer_stride: 1`
  - `learning_rate: 3e-5`

---

## 7) Results (from `experiments/results/straw_1k_v2`)

### 7.1 Base vs Mixed vs STRAW

| Model | SAMSum (ROUGE-L) | Dolly (Token-F1) | CodeAlpaca (Exact-Match norm) | Macro Avg |
|---|---:|---:|---:|---:|
| Base | 0.1974 | 0.2446 | 0.0000 | 0.1474 |
| Mixed LoRA | 0.3931 | 0.3409 | 0.0500 | 0.2613 |
| **STRAW** | **0.4007** | **0.3412** | 0.0500 | **0.2640** |

Observations:
- STRAW improves over Base by a wide margin on all macro metrics.
- STRAW is slightly better than Mixed LoRA on macro average and on SAMSum/Dolly.
- CodeAlpaca is tied between STRAW and Mixed in this run.

### 7.2 Domain-Specific LoRA (single-domain upper references)

| Domain Adapter | In-domain Score |
|---|---:|
| LoRA on SAMSum | 0.4274 |
| LoRA on Dolly | 0.3619 |
| LoRA on CodeAlpaca | 0.1000 |

These remain strong in-domain references; STRAW's value is cross-domain dynamic behavior with one shared mechanism.

---

## 8) Visual Analysis of Dynamic LoRA (`B @ A`)

Artifacts:
- Heatmap/GIFs: `ba_viz_1k/`
- Domain compare maps: `ba_compare_1k/`
- 3D surfaces: `ba_fancy_3d_1k/`

### 8.1 Absolute BA maps (`ba_compare_1k/layer_*_abs.png`)
- Show per-domain effective adapter structure at each layer.
- Pattern fields are similar in global scale but differ locally, indicating domain-conditioned adaptation.

### 8.2 Difference maps (`ba_compare_1k/layer_*_diff.png`)
- Highlight pairwise domain shifts (`codealpaca-dolly`, `codealpaca-samsum`, `dolly-samsum`).
- Some pairs are near-zero in certain layers; others show stronger structured differences.

### 8.3 Frobenius norm trend (`ba_compare_1k/layer_norms.png`)
- Layer-wise BA magnitude decreases gradually with depth.
- Domain curves are close but not identical, supporting subtle domain-specific modulation.

### 8.4 3D surfaces (`ba_fancy_3d_1k/layer_*_surfaces.png` and `*_diff_surfaces.png`)
- Surfaces provide an intuitive view of local intensity and topology differences across domains.
- Difference surfaces make cross-domain contrast visually obvious even when 2D maps appear noisy.

---

## 9) Why This Is Cool (Presentation Line)

Instead of choosing one static adapter for all prompts, STRAW synthesizes adapter weights from each prompt's prefix state.  
That means adaptation happens at inference time, per sample, while keeping the base model frozen and efficient.

---

## 10) Next Improvements

- Expand target modules beyond `v_proj` (e.g., `q_proj`) for added adaptation bandwidth.
- Run larger sample settings (after 1k proof) for stronger benchmark confidence.
- Add ablations by rank/stride and show W&B sweeps in final deck.

