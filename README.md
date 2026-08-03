# LLM Wiki Notebooks

Jupyter notebooks accompanying the [llm-wiki](https://github.com/jjames/llm-wiki) knowledge base — a structured curriculum for studying ML foundations through frontier AI research.

## Learning Path

### Neural-network and language-model progression

| # | Notebook | Topics | Role |
|---|----------|--------|------|
| 01 | [`01_perceptron.ipynb`](01_perceptron.ipynb) | Perceptron, sigmoid, decision boundaries | Core |
| 02 | [`02_mlp_backprop.ipynb`](02_mlp_backprop.ipynb) | MLP, backpropagation, vanishing gradients | Core |
| 03 | [`03_cnn.ipynb`](03_cnn.ipynb) | Convolutional networks, feature maps | Optional vision branch |
| 04 | [`04_rnn.ipynb`](04_rnn.ipynb) | Recurrent networks, hidden state | Core |
| 05 | [`05_lstm.ipynb`](05_lstm.ipynb) | LSTM, gating, long-range dependencies | Core |
| 06 | [`06_seq2seq.ipynb`](06_seq2seq.ipynb) | Sequence-to-sequence, encoder-decoder | Historical bridge |
| 07 | [`07_attention.ipynb`](07_attention.ipynb) | Attention mechanism, Bahdanau attention | Historical bridge |
| 08 | [`08_transformer.ipynb`](08_transformer.ipynb) | Full transformer architecture | Core |
| 09 | [`09_gpt.ipynb`](09_gpt.ipynb) | Autoregressive language modeling, GPT | Core |
| 10 | [`10_bert.ipynb`](10_bert.ipynb) | Masked language modeling, BERT | Encoder-model branch |

### Mechanistic interpretability

| # | Notebook | Topics |
|---|----------|--------|
| 11 | [`11_superposition.ipynb`](11_superposition.ipynb) | Toy models of superposition (Elhage et al. 2022) |
| 12 | [`12_sparse_autoencoder.ipynb`](12_sparse_autoencoder.ipynb) | Sparse autoencoders, monosemanticity (Bricken et al. 2023) |

Read these alongside the wiki pages on [superposition](https://github.com/jjames/llm-wiki/blob/main/wiki/superposition.md) and [Towards Monosemanticity](https://github.com/jjames/llm-wiki/blob/main/wiki/towards-monosemanticity.md). Notebooks 11–12 run several CPU training sweeps and can take substantially longer than the earlier notebooks.

### Prerequisite guided foundations — Part II

| # | Notebook | Topics |
|---|----------|--------|
| 18 | [`18_linear_algebra_foundations.ipynb`](18_linear_algebra_foundations.ipynb) | Vector geometry, transformations, tensors/einsum, systems, bases, conditioning, least squares, SVD/PCA |
| 19 | [`19_calculus_optimization_foundations.ipynb`](19_calculus_optimization_foundations.ipynb) | Stable functions, derivatives, directional change, chain rule, Jacobians, curvature, gradient descent |
| 20 | [`20_probability_statistics_foundations.ipynb`](20_probability_statistics_foundations.ipynb) | Distributions, Bayes, covariance, Monte Carlo, CLT, intervals, MLE, Bayesian updating, power |
| 21 | [`21_information_theory_foundations.ipynb`](21_information_theory_foundations.ipynb) | Surprise, entropy, cross-entropy, KL, mutual information, data processing, Huffman coding, ELBO |

These are teaching labs for the complete four-track [prerequisite curriculum](https://github.com/jjames/llm-wiki/blob/main/lessons/00-curriculum-map.md). They introduce each idea through short derivations, visual experiments, assertions, and cumulative questions before the less-guided mastery layer.

### Prerequisite mastery labs

| # | Notebook | Topics |
|---|----------|--------|
| 13 | [`13_linear_algebra_mastery.ipynb`](13_linear_algebra_mastery.ipynb) | Centering, SVD, low-rank recovery, basis invariance, least squares |
| 14 | [`14_calculus_optimization_mastery.ipynb`](14_calculus_optimization_mastery.ipynb) | Stable softmax, manual gradients, gradient checks, GD, momentum, AdamW |
| 15 | [`15_probability_statistics_mastery.ipynb`](15_probability_statistics_mastery.ipynb) | Paired evaluation, bootstrap intervals, randomization tests, Bayesian updating |
| 16 | [`16_information_theory_mastery.ipynb`](16_information_theory_mastery.ipynb) | Entropy, mutual information, data processing, ELBO, bits-back coding |

These are executable labs for the four-track [prerequisite mastery layer](https://github.com/jjames/llm-wiki/tree/main/mastery). Each includes assertions, investigation prompts, and a handoff to its capstone.

### Prerequisite deep dives

| # | Notebook | Topics |
|---|----------|--------|
| 17 | [`17_autodiff_from_scratch.ipynb`](17_autodiff_from_scratch.ipynb) | Finite-difference diagnostics, dual-number JVPs, scalar reverse mode, VJPs, HVPs, XOR MLP |
| 22 | [`22_linear_systems_subspaces_conditioning.ipynb`](22_linear_systems_subspaces_conditioning.ipynb) | Four fundamental subspaces, projections, QR least squares, conditioning, pseudoinverse, ridge |
| 23 | [`23_multivariable_calculus_optimization_geometry.ipynb`](23_multivariable_calculus_optimization_geometry.ipynb) | Directional derivatives, Taylor models, Hessian geometry, GD stability, constrained optimization |
| 24 | [`24_statistical_inference_calibration_model_checking.ipynb`](24_statistical_inference_calibration_model_checking.ipynb) | Coverage, power, likelihood geometry, calibration, posterior predictive model checking |

Notebook 17 expands the calculus track's matrix-calculus and autodiff module into a from-scratch implementation lab. Notebooks 22–24 strengthen the most important gaps between the compact guided surveys and independent mastery work. Complete the applicable deep dive before its mastery lab when the topic is new; use it afterward as targeted repair when a diagnostic reveals the gap.

### Recommended prerequisite notebook order

| Track | Guided foundations | Deep dive | Mastery lab |
|---|---|---|---|
| Linear algebra | 18 | 22 (systems and conditioning) | 13 |
| Calculus and optimization | 19 | 23 (optimization geometry) → 17 (autodiff) | 14 |
| Probability and statistics | 20 | 24 (inference and model checking) | 15 |
| Information theory | 21 | — | 16 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

Notebook 03 downloads MNIST into `data/` on first execution. CUDA is optional; every notebook has a CPU path.

## Validation

The repository checks notebook structure, committed error outputs, dependency coverage, the pedagogical-check sections in Notebooks 01–12, and clean-kernel execution of the fast deterministic suite:

```bash
python -m pip install -r requirements-ci.txt
python scripts/check_notebooks.py
python scripts/check_notebooks.py --execute fast
```

Use `python scripts/check_notebooks.py --execute all --timeout 1800` for the full suite after installing `requirements.txt`. The full run downloads MNIST and includes the long interpretability experiments.

Committed outputs are optional because some visualizations are useful when browsing on GitHub. Error outputs are never allowed, and any edited notebook should be rerun from a clean kernel before it is merged.

## Exploratory Material

Early calculus and scalar-autograd scratchpads live in [`scratch/`](scratch/README.md). They are retained as provenance for Notebook 17, but they are not part of the curriculum or the execution suite.
