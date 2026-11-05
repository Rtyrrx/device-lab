# Browser application

This directory is the complete static deployment for Device Lab. It runs two trained classifiers directly in the browser; no Python process, inference server, or API key is required.

## Local preview

```sh
python -m http.server 8080 --directory phone-site
```

Run the command from the repository root and open [http://localhost:8080](http://localhost:8080). Camera capture is available on compatible phones through an image file input with `capture="environment"`. HTTPS hosting is recommended for phone use. Desktop browsers generally show a file picker.

## Architecture

`index.html` loads the local TensorFlow.js library, WebAssembly backend, and `assets/app.mjs`. The application initializes the backend on first analysis, downloads the model manifests and weight shards, and runs both models on a resized RGB image.

- `assets/app.mjs`: image selection/capture, preprocessing, simulated conditions, rendering, and session timing.
- `assets/inference.mjs`: model loading, float16 decoding, MobileNetV2 and ConvNeXt execution, and CAM computation.
- `models/mobilenet.json` and `models/convnext.json`: architecture metadata and shard manifests.
- `models/*.bin`: trained weight shards, each at most 4 MiB.
- `assets/metrics.json`: recorded Python experiment results displayed independently of browser timings.
- `assets/palette.json`: heatmap colors.

Model shaders/kernels are provided by TensorFlow.js. The application uses the WebAssembly backend with one runtime thread for broad compatibility. A modern browser and enough memory to hold the larger ConvNeXt model are required.

## Model export and validation

`../export_phone_models.py` exports the trained local Keras checkpoints without retraining. Parameters are stored as little-endian float16, then decoded to float32 tensors for runtime execution. Manifest entries record tensor shapes, offsets, shard sizes, and SHA-256 checksums. Downloaded shards are checked when the browser's cryptographic API is available and cached when storage permits.

The implementation reproduces MobileNetV2 preprocessing and graph execution, ConvNeXt normalization and blocks, both trained classifier heads, and Class Activation Mapping. CAM weights each spatial feature channel by the selected class's linear classifier weight, retains positive evidence, normalizes the result, and overlays it on the image.

The conversion check compared both models with Python on all 60 original clean test images. Both matched all predicted labels. Numeric confidence differences and individual comparisons are recorded in [../results/browser_full_parity.json](../results/browser_full_parity.json). This check does not establish equivalent behavior on every input or hardware configuration.

The results table contains recorded desktop metrics. Browser-side blur, noise, and other effects are approximate counterparts of the Python experiment transforms. Session inference timing excludes downloads, photo decoding, and heatmap rendering.

## Functional self-check

Append `?selfcheck=1` to the application URL to expose **Use generated test pattern**. This creates a synthetic gradient inside the browser and exercises the same image-selection, prediction, and rendering flow without opening a private file. It is a functional check, not a recognition-accuracy test. The control is hidden during normal use.

## Deployment

`.github/workflows/deploy-pages.yml` publishes this directory to GitHub Pages on changes to the site or deployment workflow on `main`. It can also be run manually with `workflow_dispatch`.

The workflow uses `contents: read`, `pages: write`, and `id-token: write` for checkout and Pages deployment. Configure the repository's Pages source as **GitHub Actions**. Only this directory is included in the deployment artifact.

After changing application modules or styles, update their version query parameters in `index.html` to prevent stale browser caches. Changes to model shards require updated sizes and checksums in the manifests; the Python exporter generates these values.

## Privacy and limitations

Photos remain in the browser. The site contains no original training/test photographs as sample assets. Public model downloads and normal hosting requests still occur.

The classifiers always select among monitor, telephone, laptop, and keyboard. Scores are uncalibrated, and unsupported objects can receive confident incorrect labels. Heatmaps show positive class evidence, not exact object outlines or proof of correctness. Phone camera controls and available memory vary across devices.

TensorFlow.js 4.22.0 is distributed under Apache-2.0; see [assets/TENSORFLOWJS-LICENSE.txt](assets/TENSORFLOWJS-LICENSE.txt).
