# MNIST MLP (PyTorch)

Small, readable PyTorch project that trains a multi-layer perceptron on MNIST. Useful as a portfolio-ready reference for data loading, training, evaluation, and quick inference.

## Highlights
- Minimal, well-commented code paths for training (`train.py`) and inference (`inference.py`)
- Configurable hyperparameters in one place (`config.py`)
- Dataset download handled automatically via `torchvision`
- Saves a reusable checkpoint to `mnist_mlp.pth` after training

## Project Layout
- `config.py` — batch size, learning rate, epochs, hidden units, and normalization stats
- `data.py` — MNIST transforms plus train/test data loaders
- `model.py` — simple fully connected network for 28×28 grayscale digits
- `train.py` — training loop, evaluation, loss curves, and checkpoint saving
- `inference.py` — loads `mnist_mlp.pth` and predicts a sample digit
- `data/` — MNIST data will be downloaded here automatically
- `mnist_mlp.pth` — sample trained weights (regenerated when you rerun training)

## Setup
1. Install Python 3.9+ and `pip`.
2. (Recommended) Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```

## Run Training
From the repo root:
```bash
python train.py
```
- Prints batch-level loss during training and final test accuracy.
- Displays epoch and batch loss curves.
- Writes the checkpoint to `mnist_mlp.pth`.

## Run Inference
After training (or using the included checkpoint):
```bash
python inference.py
```
- Loads the saved weights.
- Runs a forward pass on a sample from the test set and prints the true and predicted labels.

## Customize
- Tweak hyperparameters in `config.py` (e.g., `hidden_size`, `learning_rate`, `num_epochs`, `batch_size`).
- The transforms in `data.py` define normalization and can be adjusted for experiments.
- Swap the model architecture inside `model.py` if you want to try different layer sizes or activations.

## Notes
- MNIST downloads to `./data` on first run; no manual steps needed.
- Training is CPU-friendly for quick experiments; enabling GPU is automatic if PyTorch is installed with CUDA and your environment supports it.
- The code is concise on purpose—ideal for interviews, walkthroughs, or as a starting point for more advanced experiments.
