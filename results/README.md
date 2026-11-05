# Experiment artifacts

This directory contains recorded metrics and validation data for the four-class device classifier. Original dataset images and image-bearing development previews are not distributed.

## Files

| Artifact | Contents |
| --- | --- |
| `metrics.json` | Protocol, environment, per-condition accuracy/F1, confusion matrices, per-class metrics, and desktop latency |
| `comparison.csv` | Compact model-by-condition results |
| `predictions.csv` | Per-image labels and confidence for each model/condition |
| `plain_history.json`, `robust_history.json` | Classifier-head training histories |
| `plain_curves.png`, `robust_curves.png` | Training and validation curves |
| `confusion_matrices.png`, `robustness.png` | Summary visualizations |
| `split_manifest.json` | Original split paths and SHA-256 image hashes |
| `near_duplicate_candidates.json` | Cross-split perceptual-similarity candidates |
| `duplicate_exclusions.json` | Reviewed training/validation exclusions and effective counts |
| `browser_full_parity.json` | Python-versus-WebAssembly predicted-label and confidence comparison for both models on 60 clean images |

Model IDs are stable experiment identifiers: `a1_mobilenet` is the MobileNetV2 baseline, `convnext_plain` uses original training examples, and `convnext_robust` adds augmented training examples. The `robust` identifier names the training recipe; it is not a claim of superior performance in every condition.

## Protocol

The original split has 280 training, 60 validation, and 60 test images across four balanced classes. A reviewed monitor example is excluded from training and a telephone example is excluded from validation, leaving 279/59 effective fitting/validation images. The original 60-image test set is unchanged.

ConvNeXt-Tiny is an ImageNet-pretrained frozen feature extractor. Only a 768-to-4 softmax classifier head is trained. Both classifier runs use seed 42, the same initial weights, Adam at 0.001, batch size 32, clean-validation cross-entropy checkpoint selection, and up to 80 epochs with patience 10. The augmented run adds two deterministic variants per retained training image, for 837 examples in total. This ablation changes training-set size and updates per epoch; it is not compute-matched.

Each model is evaluated on clean images and fixed dim, blur, noise, and central-occlusion variants. These variants remain correlated versions of the same 60 images. Macro-F1 weights each class equally. Each error changes test accuracy by approximately 1.67 percentage points.

Desktop latency uses batch one, three warmups, and 20 measured runs. It excludes loading, image I/O, explanations, and browser rendering. ConvNeXt uses compiled feature extraction in this benchmark. The browser app measures its own runtime separately.

## Interpretation

All models achieved 100% on the small clean test set. The newer architecture improved some corruption conditions and performed worse on blur. Adding the selected augmentation recipe reduced ConvNeXt's descriptive mean stress accuracy by approximately 0.83 percentage points in this run. A single seed and test split do not establish statistical significance or broad real-world robustness.

The browser conversion retained all 60 predicted labels for each model. Float16 weight storage, float32 runtime operations, browser preprocessing, and hardware differences can affect confidence and predictions on other inputs. The maximum confidence differences observed in the recorded check were 0.00333 for MobileNetV2 and 0.000403 for ConvNeXt.

## Reproduction limits

`train_evaluate.py` regenerates experiment outputs when the original local dataset and checkpoint artifacts are available. `verify_dataset.py` checks the recorded hash manifest. Feature caches must be invalidated when the input data, transforms, or backbone changes.

The original web-image dataset is excluded because a complete source/license manifest is unavailable. The hashes and relative paths support integrity checks but do not supply the images. Exact cross-split duplicates were not found; reviewed similarity screening is not exhaustive, and ImageNet pretraining overlap has not been ruled out.
