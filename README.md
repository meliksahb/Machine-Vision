# Machine-Vision
Machine Vision project

# RF-DETR Experimental Framework 🛰️

A lightweight, notebook-centred test-bed for **Rapid-Fine-Tuned DETR (RF-DETR)** and a set of classical computer-vision (CV) augmentations that together form an *“improved”* two-stage detector.  
The notebook lets you

* run the **baseline large RF-DETR** model on arbitrary images or a webcam,
* apply a set of **test-time augmentations (TTA)** and a *simple* non-maximum suppression (NMS) to merge predictions (the “improved version”),
* benchmark both pipelines on
  * built-in toy test images,
  * automatically generated *challenging* images (low light, motion-blur, occlusion, scale extremes…),
  * your own folders or live camera frames, and
* try a **fundamental-CV back-up detector** (colour/edge/contour cues only, no deep nets).

---

## 1. Quick-start

### 1.1 Run interactively (recommended)

```bash
# Clone or download this folder
conda create -n rf_detr python=3.10 -y
conda activate rf_detr
pip install -r requirements.txt   # or run !pip install … inside the notebook
jupyter lab RF_Detr_and_Improved_Version.ipynb
```

## 2. Folder Layout

.
├── RF_Detr_and_Improved_Version.ipynb
├── requirements.txt
├── rf_detr_experiments/          # auto-created; predictions & visualisations
│   ├── standard_*.jpg
│   ├── challenging_*.jpg
│   └── camera_*.jpg
└── results.zip                   # zipped copy of the folder (auto-generated)

## 3. Core Classes and Functions

| Component                                     | Purpose                                                                            |
| --------------------------------------------- | ---------------------------------------------------------------------------------- |
| `RFDETRExperimentalFramework`                 | Thin façade that owns *both* RF-DETR checkpoints and CV utilities.                 |
| `test_rf_detr_original(images)`               | Runs **large** RF-DETR with a single confidence threshold (0.4).                   |
| `implement_rf_detr_improvements(images)`      | Performs TTA (horizontal flip + two extra thresholds) → merges with `_simple_nms`. |
| `_calibrate_confidence(det, weight)`          | Post-hoc scaling of logits when combining augmentations.                           |
| `_simple_nms(boxes, scores, iou_th=0.5)`      | Minimal IoU-based suppression (no class awareness).                                |
| `implement_fundamental_cv_techniques(images)` | Pure-CV fall-back: colour segmentation, Canny+contours, MSER blobs.                |
| `create_challenging_dataset()`                | Makes synthetic edge-case images (blur, darkness, extreme scale, tilt, etc.).      |
| `load_camera_images(num_frames, delay)`       | Grabs live frames via OpenCV if a webcam is present.                               |

## 4. Improved Version

| Stage             | Baseline RF-DETR (Large)                              | Improved Ensemble                                                                |
| ----------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------- |
| Model             | Single forward pass at `thr=0.40`.                    | Same backbone **+** two lower thresholds (0.30, 0.50) **+** a flipped inference. |
| Fusion            | n/a                                                   | Predictions from 4 passes are merged with `_simple_nms` (IoU ≤ 0.5).             |
| Score calibration | Raw sigmoid conf.                                     | Linear down-weight (`β`) for augmentations to avoid high-confidence clones.      |
| Result            | Faster, but may miss low-contrast or partial objects. | Slightly slower, noticeably ↑ recall in low-light & occlusion tests.             |

## 5. Try Experiments

Place evaluation images in ./Images/ or modify folder_path at the bottom cell.

Run all cells.

The script spits colour-coded PNGs with class labels and confidences into ./rf_detr_experiments/.

A zip archive (results.zip) is created for one-click download / sharing.

## 6. Extending the framework

New augmentations – Add new branches inside implement_rf_detr_improvements, append to augmented_predictions.

Different backbones – Replace RFDETRBase / RFDETRLarge stubs with your own HuggingFace model; keep .predict(image, threshold) API.

Metrics – Plug in torchmetrics or supervision.metrics inside evaluate_predictions() (left as an exercise).

Training – The notebook is inference-only; see the original RF-DETR repo for training scripts.


## 7. Troubleshooting

| Symptom                       | Fix                                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: rfdetr` | The PyPI build occasionally lags; run `pip install git+https://github.com/charlesShang/RFDet.git` |
| Webcam cell hangs             | Ensure `/dev/video0` exists (Linux) or change the device index in `cv2.VideoCapture(0)`.          |
| CUDA out of memory            | Switch to `RFDETRBase()` (smaller), or export `TORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`.       |
| Over-suppression by NMS       | Lower `iou_th` in `_simple_nms`; annotate small objects with high overlap carefully.              |

