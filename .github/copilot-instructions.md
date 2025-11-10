## Purpose
Brief, actionable guidance for an AI coding agent working on this MNIST image-classifier project.

## Quick start (how this repo is run)
- Primary runnable script: `main.py` (root of repository). Running `python main.py` will:
  - download MNIST via `torchvision.datasets.MNIST` into `./data`
  - preprocess the whole MNIST train/test sets in memory using `preprocess_dataset`
  - train `DigitClassifier` for a small number of epochs and save `digit_model.pth`
  - evaluate accuracy on the preprocessed test set

## Key files and symbols to reference
- `main.py` — single-file project. Important functions/classes:
  - `binarize(img, threshold=128)` — converts grayscale -> binary (foreground=1)
  - morphological ops: `erosion`, `dilation`, `opening`, `closing` and `B` (3×3 structuring element)
  - `preprocess_image_np(np_img)` — preprocessing pipeline used for both train & test (calls `opening` then resizes to 28×28, applies `transforms.ToTensor()` and `Normalize((0.5,), (0.5,))`)
  - `preprocess_dataset(dataset)` — converts a torchvision dataset into stacked tensors (loads entire dataset into memory)
  - `DigitClassifier(nn.Module)` — model definition (Conv -> Conv -> Linear head)
  - `predict_digit(image_path)` — convenience function to predict a single custom image

## Data layout and gotchas
- There are two apparent data sources in the repository:
  - local folders `training/` and `testing/` containing subfolders `0/`..`9/` (image folders). These are NOT referenced by `main.py` currently.
  - `./data` used by `torchvision.datasets.MNIST` (this is what `main.py` downloads and uses).

- Important: `preprocess_dataset` eagerly loads and stacks all images into memory via `torch.stack(...)`. This is fine for MNIST (small), but avoid the same pattern for large datasets.

## Dependencies (discoverable from imports)
- Python packages required: `torch`, `torchvision`, `numpy`, `scipy`, `Pillow` (PIL). Confirm versions in environment when making changes.

## Project-specific conventions & patterns
- Preprocessing is implemented in NumPy + SciPy morphological ops then converted back to PIL/torch tensors. Keep all image preprocessing changes centralized in `preprocess_image_np`.
- Normalization uses `Normalize((0.5,), (0.5,))` and the model expects single-channel inputs (shape `[1,28,28]`).
- Model weights are saved with `torch.save(model.state_dict(), 'digit_model.pth')`. Loading code should use `model.load_state_dict(torch.load(path))` and set `model.eval()` for inference.

## Typical edits an agent may be asked to make
- Change preprocessing: update `preprocess_image_np` and keep the binary morphology (use `B` and `opening`) in place unless replacing it intentionally.
- Add GPU support: current code does not use `torch.device`. If adding CUDA, move model and tensors to device consistently (eg. `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).
- Replace eager preprocessing with an on-the-fly `Dataset`/`DataLoader` if memory or speed becomes an issue — but keep tests passing and reference existing `preprocess_dataset` when migrating.

## Integration points & outputs
- Model artifact: `digit_model.pth` is produced in repo root after training.
- Single-image inference via `predict_digit(image_path)` — useful for quick checks. It uses the same preprocessing pipeline as training.

## Tests / CI
- There are no automated tests or CI configs in the repository. If asked to add tests, target small unit tests around `preprocess_image_np` (input shape and output tensor range) and a smoke test for `DigitClassifier` forward pass.

## When in doubt — specific places to read first
- Start with `main.py` top-to-bottom. For preprocessing changes, edit `preprocess_image_np` and `B`.
- For model architecture/hyperparams, edit `DigitClassifier` and the `epochs`, `optimizer` and `criterion` declarations.

## Question prompts for the repo owner
- Should local `training/` and `testing/` image folders be wired into the training script? If yes, an agent should add a Dataset that reads from those folders instead of `datasets.MNIST`.

If anything here is unclear or incomplete, tell me what area you'd like expanded (data sources, preprocessing, GPU support, or adding tests) and I'll iterate.
