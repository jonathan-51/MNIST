# MNIST Digit Classifier

A from-scratch 2-layer MLP trained on the MNIST handwritten digit dataset, built using NumPy only.

**Architecture:** Input (784) → Hidden Layer (64, ReLU) → Output (10, Softmax)

## Files

| File | Description |
|---|---|
| `MINST-2.0.py` | Main training script — training, validation, logging, LR finder, temperature scaling |
| `MINST.py` | Original implementation, kept as reference |
| `MINST.test.py` | Inference on the held-out test set |
| `graphing.py` | Plots loss, accuracy, calibration curves, and confusion matrices |
| `Convolution/` | Convolutional model — incomplete |

## Requirements

```
numpy, idx2numpy, matplotlib, pandas, seaborn
```

## Dataset

Place the MNIST `.idx` files in the root directory. Available from [Wikiwand](https://www.wikiwand.com/en/articles/MNIST_database).
