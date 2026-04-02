# Neural Networks and Deep Learning

A short course with code covering neural network fundamentals, from linear algebra foundations through modern architectures (CNNs, RNNs, Transformers, GANs, Autoencoders, GCNs).

## Course Structure

### Notebooks

| Section | Topics | Framework |
|---------|--------|-----------|
| **Linear Algebra** | Vectors, matrices, eigendecomposition, SVD, PCA | NumPy |
| **Numpy** | Preprocessing, regression, neural networks from scratch, GCNs | NumPy |
| **Probability** | PMF/PDF, marginal & conditional probability | NumPy |
| **PyTorch** | ANN, CNN, RNN/LSTM/GRU, GAN, Autoencoder, Transformers, Style Transfer, Transfer Learning | PyTorch |
| **TensorFlow** | ANN, CNN, RNN/LSTM, GAN, Autoencoder, GCN, Style Transfer, Transfer Learning | TensorFlow/Keras |
| **Comparison** | PyTorch vs TensorFlow side-by-side (NLP, classification pipelines) | PyTorch + TensorFlow |
| **Reinforcement Learning** | Auction simulation with neural networks | TensorFlow/Keras |

### Setup

```bash
pip install -r requirements.txt
```

For PyTorch with GPU support, follow [pytorch.org](https://pytorch.org/get-started/locally/).

For optional dependencies (geospatial, graph neural networks, etc.), see comments in `requirements.txt`.

### Data

Datasets are in the `data/` directory or downloaded automatically by the notebooks (MNIST, CIFAR10, etc.).

## Books

+ [Zhang, Aston, et al. "Dive into deep learning." arXiv preprint arXiv:2106.11342 (2021).](http://d2l.ai/index.html)
+ [Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
+ [Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.](https://www.deeplearningbook.org/)
+ [Hastie, Tibshirani and Friedman. The Elements of Statistical Learning. 2nd edition, 2009](https://web.stanford.edu/~hastie/ElemStatLearn/)
+ [Molnar, Christoph. Interpretable Machine Learning. Lulu. com, 2020.](https://christophm.github.io/interpretable-ml-book/)
+ [Strang, Gilbert. Linear algebra and learning from data. Wellesley-Cambridge Press, 2019.](https://math.mit.edu/~gs/learningfromdata/)

## Papers

+ [100 best NLP Papers](https://github.com/mhagiwara/100-nlp-papers)
+ [Papers on Text Style Transfer](https://github.com/fuzhenxin/Style-Transfer-in-Text)
+ [Papers with Code](https://paperswithcode.com/)

## Compatibility Notes

All notebooks were updated in April 2026 for compatibility with current library versions:

| Library | Minimum Version | Notes |
|---------|----------------|-------|
| Python | 3.10+ | |
| NumPy | 1.24+ | `np.float`/`np.int` removed in 2.0 |
| Pandas | 2.0+ | `DataFrame.append()` removed |
| Seaborn | 0.13+ | `distplot()` removed in 0.14 |
| Matplotlib | 3.6+ | Seaborn style names changed |
| NetworkX | 3.0+ | `nx.info()`, `nx.attr_matrix()` removed |
| TensorFlow | 2.16+ | Uses Keras 3 by default |
| PyTorch | 2.0+ | `Variable` removed, `pretrained=` deprecated |
| Scikit-learn | 1.3+ | |

Some notebooks have compatibility notices for archived libraries (StellarGraph, AllenNLP, torchtext legacy API) — see the notices in each notebook for alternatives.
