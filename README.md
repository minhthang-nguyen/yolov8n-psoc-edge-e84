# YOLOv8n Deployment on PSoC™ Edge E84

Embedded deployment of an INT8 YOLOv8n object detector on the **Infineon PSoC™ Edge E84 AI Kit** for chrysanthemum growth-stage detection.

This repository focuses on the deployment side of the project: model integration, INT8 input/output handling, memory placement, YOLOv8-specific post-processing, and on-device performance evaluation.

## Project overview

The model detects four chrysanthemum growth stages:

- `ThanLa` — vegetative stage
- `RaNu` — budding stage
- `NoHoa` — flowering stage
- `Harvest` — harvest stage

The deployed model uses a **320 × 320 × 3** image input and produces the YOLOv8 raw output layout with 8 channels:

```text
[x, y, w, h, class_0, class_1, class_2, class_3]
```

Unlike YOLOv5-style outputs, this YOLOv8 output does not use a separate objectness channel. The embedded post-processing therefore selects detections directly from class confidence scores.

## Deployment pipeline

```text
Camera / image input
        ↓
320 × 320 × 3 uint8 image
        ↓
INT8 input conversion
        ↓
YOLOv8n INT8 inference
        ↓
Arm® Ethos™-U55 NPU
        ↓
INT8 raw output [8 × 2100]
        ↓
Dequantization to float
        ↓
Confidence filtering
        ↓
Class-wise Non-Maximum Suppression
        ↓
Up to 5 final detections
```

## Hardware and software

| Item                  | Configuration                                 |
|-----------------------|-----------------------------------------------|
| Target board          | Infineon PSoC™ Edge E84 AI Kit                |
| NPU                   | Arm® Ethos™-U55                               |
| Model                 | YOLOv8n                                       |
| Input                 | 320 × 320 × 3                                 |
| Quantization          | INT8                                          |
| Raw output            | 8 × 2100                                      |
| Programming language  | C                                             |
| Runtime               | Infineon MTB ML / TFLM deployment stack       |

## Model memory configuration

The deployment source defines:

| Memory region                 | Size                  |
|-------------------------------|-----------------------|
| Working buffer                | 324,000 bytes         |
| Model state / tensor arena    | 3,101,712 bytes       |
| Read-only model section       | 2,808,352 bytes       |

Runtime working memory is placed in SoCMEM, while the read-only model parameters are stored in Flash.

## On-device results

| Metric                                | YOLOv8n INT8  |
|---------------------------------------|---------------|
| mAP50                                 | 83.1%         |
| Inference + post-processing latency   | 233 ms/frame  |
| Approximate throughput                | 4.29 FPS      |
| Average power                         | 0.97 W        |
| Energy per detection                  | 0.226 J       |
| RAM footprint                         | ~3.27 MiB     |
| Flash footprint                       | ~2.68 MiB     |

These results correspond to the evaluated PSoC™ Edge E84 deployment reported in the associated research work.

## Repository structure

```text
.
├── README.md
├── .gitignore
├── CODE_REVIEW.md
├── THIRD_PARTY_NOTICES.md
├── docs/
│   └── benchmark.md
└── src/
    ├── README.md
    └── model_yolov8n.c
```

## Important source-code note

The current `model_yolov8n.c` in this repository is the deployment implementation supplied for portfolio documentation.

If the embedded model-weight array is intentionally omitted, the source file is **not standalone-buildable** until the actual model binary/C-array is restored. This is documented explicitly rather than presenting the repository as a complete reproducible firmware image.

## Key implementation sections

Reviewers interested in the embedded-AI contribution can look for:

- `cast_u8_to_i8(...)` / quantization helpers — input preparation
- `mtb_model_raw(...)` — model inference
- `dequantize_d8_to_f32(...)` — output dequantization
- `detectionfilter_yolov8_f32(...)` — YOLOv8 filtering and class-wise NMS
- `IMAI_compute(...)` — end-to-end inference pipeline
- `IMAI_init(...)` — model initialization and tensor-arena setup

See [`CODE_REVIEW.md`](CODE_REVIEW.md) for notes on the current implementation.

## Associated publication

T. M. Hai, N. M. Thang, C. M. Quang, N. H. Cuong, D. T. Hoang, N. T. Ngan, and B. D. Thanh,  
**“Deployment of YOLOv8n for Chrysanthemum Growth Stage Detection on an Edge AI Device,”**  
*Journal of Measurement, Control and Automation*, vol. 30, no. 3, pp. 70–81, 2026.  
DOI: 10.64032/mca.v30i3.448

## Base deployment framework

The embedded work uses the Infineon PSoC™ Edge E84 / DEEPCRAFT™ deployment ecosystem. The official Infineon vision deployment example is:

https://github.com/Infineon/mtb-example-psoc-edge-ml-deepcraft-deploy-vision

## Licensing and third-party material

This project uses third-party development tools, libraries, and deployment components. Their original license and copyright terms remain applicable.

No repository-wide MIT/Apache/GPL license is applied to third-party material. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Author

**Nguyen Minh Thang**  
Hanoi University of Science and Technology (HUST)

Research interests: Computer Vision, Edge AI, Embedded AI, Efficient AI, and AIoT.
