# How to run

The scripts are in `src/` and are run from that folder. Everything they write goes to
`checkpoints/`, `results/` and `splits/` in the repository root. Locally: the setup in `README.md`
(Python 3.13, `requirements.txt`, dataset). The CNNs need a GPU; a free Colab T4 is enough.

## In Colab (recommended)

1. Open `notebooks/colab_run_all.ipynb` in Colab (*File → Open notebook → GitHub*, or upload it).
2. *Runtime → Change runtime type → T4 GPU*.
3. Check `REPO_URL` (the clone address, ending in `.git`) and `BRANCH` in the first cell, then
   *Runtime → Run all*.

Checkpoints and results are saved in `MyDrive/ikt466` on Google Drive. If Colab disconnects,
run all cells again: finished training runs are skipped.

**Private repository:** create a GitHub fine-grained personal access token with read access to
the repository and use `REPO_URL = "https://<TOKEN>@github.com/OWNER/REPO.git"`. Do not save or
commit the notebook with the token in it.

## Step by step

| Step | Command (from `src/`) | Output |
|---|---|---|
| Data | `git clone https://github.com/ml4py/dataset-iiit-pet.git data/oxford-iiit-pet` at commit `ebb5f9e` (as in the README; the notebook does this) | `data/oxford-iiit-pet/` |
| Splits | `python save_splits.py` | `splits/train.csv`, `val.csv`, `test.csv` |
| Test images, grey fill | `python make_background_test.py` | `data/background_test/` |
| Test images, noise fill | `python make_background_test.py --fill noise --out ../data/background_test_noise` | `data/background_test_noise/` |
| SVM baseline | `python train_svm.py` and `python train_svm.py --color` | `checkpoints/svm_hog*.joblib`, `results/svm_gridsearch*.csv` |
| CNNs | `python train_cnn.py --model pet_cnn --seed 0` (seeds 0, 1, 2; also `--model resnet50`) | `checkpoints/<model>_seed<n>.pt`, `results/history/<model>_seed<n>.json` |
| Evaluation, grey | `python evaluate_svm_background.py` and `python evaluate_cnn_background.py` | `results/experiment2_{svm,cnn}_{summary,predictions}.csv` |
| Evaluation, noise | the same two with `--data ../data/background_test_noise --tag _noise` | `results/experiment2_{svm,cnn}_noise_*.csv` |
| Grad-CAM | `python gradcam_share.py` | `results/gradcam_per_image.csv`, `gradcam_summary.csv`, `figures/gradcam_examples.png` |
| Summary | `python summarize_results.py` | `results/main_table{,_noise}.csv/.tex`, `model_comparison{,_noise}.csv`, `gradcam_vs_accuracy.csv`, `figures/accuracy_by_variant{,_noise}.png` |
| Learning curves | `python plot_learning_curves.py` | `results/figures/learning_curves.png` |
| SVM error examples | `python analyze_svm.py` (add `--color` for the colour model) | `results/figures/svm*_examples*.png` |

`train_cnn.py --debug` trains on 64 images as a quick check. Its files end in `_debug` and are
never evaluated.

## Settings

- Split: stratified 80/20 split of the official trainval set (`SPLIT_SEED = 42` in `data.py`);
  the official test set is used for testing only.
- CNN settings: `DEFAULTS` in `train_cnn.py`. Each run's settings, learning curve, best epoch,
  training time and GPU are saved in `results/history/`.
- Exact package versions: `requirements.txt` for local runs; for the Colab runs,
  `results/environment.txt` (written by the notebook, which uses Colab's preinstalled packages).

## Checkpoints in the submission

`.gitignore` excludes `checkpoints/`, `*.pt` and `data/`. A ResNet-50 checkpoint is about 95 MB,
close to GitHub's 100 MB limit per file, so put `resnet50_seed0.pt` (and the other checkpoints) on
Google Drive or OneDrive and link them here.
