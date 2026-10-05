# Staged Class-Incremental Learning for Marine Plastic Waste Classification

This project evaluates a static transfer-learning baseline before running naive sequential learning and fixed-memory exemplar replay. It compares ResNet50 with EfficientNet-B2, chooses one architecture using **validation macro-F1 only**, and uses that selected architecture for the three incremental experiments.

## Dataset and labels

The provided Roboflow export has `train/` and `valid/` folders with `_classes.csv` annotation files. Class names are read directly from these CSVs; the current dataset labels are `can`, `foam`, `plastic`, `plastic bottle`, and `unknow`. Each image can have multiple positive labels, so the task is treated as multilabel classification; labels are not collapsed into a fabricated single ground truth.

The pipeline audits image integrity and duplicates, then creates deterministic iterative-multilabel-stratified splits: 70% train, 15% validation, and 15% test. Per-split label counts are written to `artifacts/dataset_report.json`; exact split membership and global label vectors are saved in `artifacts/data_splits.json`. Training uses inverse-frequency weighted sampling; validation and test are not oversampled. The discovered classes are automatically assigned to three incremental stages with approximately balanced positive-label support.

## Install and run in VS Code

Use Python 3.10 or newer. From the project root in a VS Code terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch/torchvision build from the official PyTorch selector for GPU training. The program automatically selects CUDA when available; otherwise it uses CPU.

Run the full pipeline:

```powershell
python train.py
```

The order is deliberate:

1. Audit the data and write train/validation/test membership.
2. Train both static backbones and compare validation macro-F1.
3. Select the higher validation-Macro-F1 architecture (test labels are not used for model selection).
4. Evaluate the selected static checkpoint on the held-out test split.
5. Run static, naive sequential, and memory-replay experiments with the selected architecture.
6. Evaluate final experiment checkpoints on test data and write plots, metrics, confusion matrices, replay exemplars, and Grad-CAM examples.

To run the two phases separately:

```powershell
python train.py --phase baseline
python train.py --phase incremental
```

The incremental phase requires baseline comparison/checkpoint files from the baseline phase and repeats the deterministic data split with the same seed. Other options:

```powershell
python train.py --epochs 5 --batch-size 16 --memory-size 100
python train.py --data-root "C:\path\to\dataset" --output-dir artifacts
python train.py --no-pretrained
python train.py --freeze-last-block
```

Defaults are seed 42, two epochs per fit, memory capacity 100, and ImageNet pretrained weights. First use downloads torchvision weights.

## Evaluation definitions

- **Accuracy** is label-wise binary accuracy across every evaluated image and every seen class.
- **Exact-match accuracy** is also reported separately: all active labels for an image must match.
- **Precision, recall, and Macro-F1** are calculated for the positive outcome of each active label, then macro-averaged over active labels using a fixed sigmoid threshold of 0.5. This definition remains consistent when a stage has only one active label.
- Each label has its own 2x2 confusion matrix in `[[true negative, false positive], [false negative, true positive]]` order, keyed by the matching class name and emitted only for classes evaluated in that stage.
- **Old-class retention** is macro recall over classes learned before the current stage. It is bounded from 0 to 1; Stage 1 is null/not applicable.
- **Forgetting** is the mean positive drop in per-class recall from each old class's best earlier stage recall. It is bounded from 0 to 1.
- Static architecture selection uses validation Macro-F1. The held-out test split is used only for final reporting.

Replay memory is fixed-capacity and class-balanced. A stage trains on new-class images and, for the proposed method, stored old-class exemplars. Loss masks supervise only labels known for the corresponding data in that stage. Results are saved as observed; the pipeline does not guarantee that replay will outperform naive learning on every run.

## Artifacts

- `artifacts/dataset_report.json`, `artifacts/data_splits.json`: dataset audit and exact split manifest.
- `artifacts/baseline_comparison.json`: validation Macro-F1 for ResNet50 and EfficientNet-B2 and selected model.
- `artifacts/baseline_test_metrics.json`, `artifacts/test_metrics.json`: held-out test metrics, separate from validation-based selection.
- `artifacts/models/`: architecture baselines, selected baseline, and per-stage incremental checkpoints.
- `artifacts/memory/`: exemplar file paths and labels at each replay stage.
- `artifacts/metrics.json`, `artifacts/metrics.csv`: stage-wise validation metrics for all three methods.
- `artifacts/confusion_matrices/`: per-class 2x2 confusion matrices as CSV and PNG.
- `artifacts/graphs/`: stage-wise accuracy, Macro-F1, retention, and forgetting.
- `artifacts/gradcam/`: one validation-image Grad-CAM overlay per class for the selected static baseline.
- `artifacts/comparison_summary.json`: final-stage naive-versus-replay forgetting and retention.

## Streamlit application

After the baseline phase, the app can use the selected static model. After incremental training it selects the newer proposed replay model automatically.

```powershell
streamlit run app.py
```

Upload a JPG, JPEG, PNG, or WEBP image to see the highest-scoring dataset label, confidence, scores for all labels, and its Grad-CAM overlay.

## Reproducibility and attribution

Python, NumPy, and PyTorch seeds are fixed; cuDNN deterministic mode is enabled. Hardware, CUDA, and library versions can still cause numerical variation.

The dataset README identifies [Roboflow Universe: Marine Debris](https://universe.roboflow.com/datavision-2dcwk/marine-debris-i2ge3-ftwk4) as the source and states CC BY 4.0. Preserve dataset attribution and license terms when redistributing it.
