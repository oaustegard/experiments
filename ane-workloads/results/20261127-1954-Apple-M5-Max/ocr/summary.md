# Apple Vision OCR on Omni Parse Bench: Apple M5 Max, 2026-11-27T19:54:23

macOS 27.2.1, AC True, low power False. 250 pages (0 could not be rasterised).

| arm | GPU | pages | ms/page median | mean | p90 | pages/s | idle median, same pages | errors | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|---|---|
| text | idle | 250 | 184.9 | 202.7 | 308.7 | 4.93 |  | 0 |  |
| text | busy | 150 | 245.9 | 263.2 | 379.8 | 3.8 | 179.1 | 0 | 45.05 -> None (busy 1.0) |
| docs | idle | 250 | 308.7 | 331.5 | 480.0 | 3.02 |  | 0 |  |
| docs | busy | 150 | 337.2 | 393.0 | 590.3 | 2.54 | 284.7 | 0 | 45.05 -> 42.92 (busy 0.727) |

- text: {"request": "VNRecognizeTextRequest", "level": "accurate", "revision": 3, "language_correction": true, "automatically_detects_language": true, "asked_device": "auto", "supported_devices": {"VNComputeStageMain": ["MLNeuralEngineComputeDevice", "MLGPUComputeDevice", "MLCPUComputeDevice"]}, "assigned_devices": {"VNComputeStageMain": "auto"}}; first request 478.56 ms
- docs: {"request": "RecognizeDocumentsRequest.revision1", "swiftc": "Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)\nTarget: arm64-apple-macosx27.2.0", "supported_devices": {"main": ["ne", "gpu", "cpu"]}, "asked_device": "auto", "assigned_devices": {"main": "auto"}, "os": "Version 27.2.1 (Build 26B131)"}; first request 333.691667 ms

Omni Parse Bench, 250 pages (headline = mean of the families):

| system | headline | text | tables | layout |
|---|---|---|---|---|
| text | 26.09 | 25.5 | 0.0 | 52.8 |
| docs | 40.84 | 22.3 | 19.6 | 80.7 |
| Indra OCR | 79.54 | 64.3 | 77.4 | 97.0 |
| Tesseract | 5.51 | 16.5 | 0.0 | 0.0 |
