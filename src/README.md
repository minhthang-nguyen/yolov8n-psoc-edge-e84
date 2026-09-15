# Source

The deployment implementation belongs in this directory.

Recommended layout:

```text
src/
├── model_yolov8n.c
└── model.h
```

## Current portfolio source

`model_yolov8n.c` is an adapted deployment implementation for YOLOv8n on PSoC™ Edge E84.

Before treating the repository as a buildable firmware project, verify that the actual model binary/C-array is present. In the supplied portfolio source, the `_K3_flash[]` model-weight array is empty.

See [`../CODE_REVIEW.md`](../CODE_REVIEW.md) for the sections that should be cleaned up before publishing.
