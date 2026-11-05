# Device Lab

[Live demo](https://rtyrrx.github.io/device-lab/)

Device Lab classifies images of **monitors, telephones, laptops, and keyboards**. It compares a MobileNetV2 baseline with a ConvNeXt-Tiny classifier and displays a Class Activation Map for the ConvNeXt prediction.

The browser demo runs entirely on the device using TensorFlow.js and WebAssembly. It supports image selection, camera capture on compatible phones, simulated image conditions, confidence scores, and inference timings. Photos are not uploaded to an inference server.

## Use the demo

1. Open the [website](https://rtyrrx.github.io/device-lab/) in a current browser.
2. Select an image or use **Take photo** on a supported phone.
3. Select an optional image condition and press **Analyze image**.

The first analysis downloads approximately 60 MB of model weights. Model files are cached when browser storage permits. Camera behavior depends on the browser and operating system; desktop browsers usually present a file picker. Older devices may have insufficient memory for ConvNeXt.

Only the four listed categories are supported. Unrelated objects can receive confident, incorrect labels. Confidence scores are not calibrated, and the heatmap is a visualization of class evidence rather than an object boundary or proof of correctness.

## Run the browser app locally

The static app and its exported model files are included in `phone-site/`. No Python ML dependencies are required to serve it.

```sh
git clone https://github.com/Rtyrrx/device-lab.git
cd device-lab
python -m http.server 8080 --directory phone-site
```

Open [http://localhost:8080](http://localhost:8080). A web server is required because the app loads model files with `fetch`; opening `index.html` directly does not work.

## Project structure

| Path | Purpose |
| --- | --- |
| `phone-site/` | Deployable browser app, inference runtime, and model shards |
| `.github/workflows/deploy-pages.yml` | GitHub Pages deployment |
| `vision.py` | Python models, transforms, and CAM implementation |
| `train_evaluate.py` | Frozen-feature training and augmentation ablation |
| `export_phone_models.py` | Export trained Python checkpoints to browser weights |
| `app.py` | Optional local Flask inference app |
| `smoke_check.py` | Python app checks using local model/data artifacts |
| `verify_dataset.py` | Dataset hash and cross-split duplicate checks |
| `results/` | Recorded experiment metrics, curves, manifests, and conversion validation |

## Python development

The Python pipeline was tested with Python 3.10 and TensorFlow 2.21.0.

```sh
python -m venv .venv
# Activate the environment using the command for the current shell.
python -m pip install -r requirements.txt
```

Python inference and training use local artifacts that are **not included** in this repository:

- `models/a1_original.keras`: the trained MobileNetV2 baseline.
- `models/convnext_backbone.weights.h5`: frozen ConvNeXt-Tiny backbone weights.
- `models/robust_head.weights.h5`: the augmented classifier head.
- `dataset/{train,validate,test}/{monitor,telephone,laptop,keyboard}/`: the original image split, required for training, verification, and sample-based checks.

With the required local artifacts available:

```sh
python app.py                 # Local inference UI at http://127.0.0.1:5055
python verify_dataset.py      # Verify the recorded dataset manifest
python train_evaluate.py      # Reproduce training and evaluation
python export_phone_models.py # Export checkpoints into phone-site/models/
python smoke_check.py         # Exercise local prediction and upload routes
```

The Flask app binds to loopback and uses the development server. It is an optional local tool, not the GitHub Pages backend. The browser demo uses the included exported weights and does not depend on these Python checkpoints or the dataset.

## Experiments and validation

The recorded experiment uses a 400-image, four-class dataset with an original 280/60/60 training/validation/test split. Two reviewed similarity candidates were excluded from classifier fitting and validation, leaving 279 training images and 59 validation images. The test set remains 60 images. The augmented classifier uses 558 additional training-only variations.

| Model | Clean | Dim | Blur | Noise | Occluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV2 baseline | 100.00% | 98.33% | 96.67% | 91.67% | 98.33% |
| ConvNeXt, original training images | 100.00% | 100.00% | 91.67% | 98.33% | 100.00% |
| ConvNeXt with augmentation | 100.00% | 100.00% | 90.00% | 96.67% | 100.00% |

These are the recorded **Python desktop results**. Corrupted images are transformed versions of the same test images, not additional independent samples. This single-seed experiment does not establish statistical significance or general superiority of either architecture.

Both browser models matched their Python counterparts' predicted labels on all 60 clean test images in the WebAssembly conversion check. Browser weights use float16 storage with float32 runtime tensors, so confidence scores can differ slightly. The maximum observed confidence difference was 0.00333 for MobileNetV2 and 0.000403 for ConvNeXt. Results can also vary by browser/device. The website's simulated transforms approximate the desktop transforms; device timings are measured separately.

Detailed protocols and artifact descriptions are in [results/README.md](results/README.md). Browser architecture, export format, and deployment details are in [phone-site/README.md](phone-site/README.md).

## Privacy and data provenance

The hosted app processes selected photos in browser memory. It downloads public model assets and makes no photo-inference API request. The optional Flask app instead processes uploads in the local Python process.

The original dataset was collected from web images without a complete source/license manifest. Dataset photographs, image-bearing responses, and private development documents are not distributed here. ImageNet pretraining overlap has not been ruled out, and duplicate screening is not exhaustive. The dataset is not required to use the browser demo.

## References and third-party assets

- Liu et al., [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), CVPR 2022.
- Sandler et al., [MobileNetV2](https://arxiv.org/abs/1801.04381), CVPR 2018.
- Zhou et al., [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150), CVPR 2016.
- [Keras ConvNeXt documentation](https://keras.io/api/applications/convnext/convnext_models/).
- TensorFlow.js 4.22.0, Apache-2.0; the bundled library license is in [phone-site/assets/TENSORFLOWJS-LICENSE.txt](phone-site/assets/TENSORFLOWJS-LICENSE.txt).

The bundled third-party license applies to the TensorFlow.js assets; it does not grant a license to the original web-image dataset or automatically license the entire project.
