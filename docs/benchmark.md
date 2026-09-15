# Benchmark

## YOLOv8n INT8 on PSoC™ Edge E84

| Metric                                | Value         |
|---------------------------------------|---------------|
| Test mAP50                            | 83.1%         |
| Inference + post-processing latency   | 233 ms/frame  |
| Approximate throughput                | 4.29 FPS      |
| Average power                         | 0.97 W        |
| Energy per detection                  | 0.226 J       |
| RAM footprint                         | ~3.27 MiB     |
| Flash footprint                       | ~2.68 MiB     |   

The model was post-training quantized to INT8 before deployment.

## Resource allocation in the deployment source

| Region                            | Bytes     |
|-----------------------------------|-----------|
| Working buffer                    | 324,000   |
| State / tensor-arena allocation   | 3,101,712 |
| Read-only model section           | 2,808,352 |

The runtime memory arrangement places working memory in SoCMEM and keeps read-only model parameters in Flash.

## Comparison with evaluated lightweight models

| Metric                    | YOLOv5n   | YOLOv5nu  | YOLOv8n   |
|---------------------------|-----------|-----------|-----------|
| mAP50 (%)                 | 82.4      | 82.2      | 83.1      |
| Latency (ms)              | 59.6      | 174.5     | 232.8     |
| Energy / detection (J)    | 0.058     | 0.169     | 0.226     |
| RAM (MB)                  | 1.19      | 3.27      | 3.27      |
| Flash (MB)                | 1.64      | 2.29      | 2.68      |

YOLOv8n achieved the highest evaluated mAP50, while YOLOv5n provided substantially lower latency and resource demand. The comparison illustrates the accuracy–efficiency trade-off on an MCU-class Edge-AI platform.
