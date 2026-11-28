# Apple Vision OCR on Omni Parse Bench: Apple M5 Max, 2026-11-27T20:06:30

macOS 27.2.1, AC True, low power False. 2937 pages (0 could not be rasterised).

| arm | GPU | pages | ms/page median | mean | p90 | pages/s | idle median, same pages | errors | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|---|---|
| text | idle | 2937 | 182.6 | 205.7 | 308.9 | 4.86 |  | 0 |  |
| text | busy | 150 | 269.7 | 314.5 | 499.5 | 3.18 | 186.2 | 0 | 45.07 -> None (busy 1.0) |
| docs | idle | 2937 | 316.4 | 339.3 | 514.0 | 2.95 |  | 0 |  |
| docs | busy | 150 | 389.7 | 434.1 | 676.3 | 2.3 | 325.0 | 0 | 45.07 -> 43.79 (busy 0.67) |

- text: {"request": "VNRecognizeTextRequest", "level": "accurate", "revision": 3, "language_correction": true, "automatically_detects_language": true, "asked_device": "auto", "supported_devices": {"VNComputeStageMain": ["MLNeuralEngineComputeDevice", "MLGPUComputeDevice", "MLCPUComputeDevice"]}, "assigned_devices": {"VNComputeStageMain": "auto"}}; first request 339.89 ms
- docs: {"request": "RecognizeDocumentsRequest.revision1", "swiftc": "Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)\nTarget: arm64-apple-macosx27.2.0", "supported_devices": {"main": ["ne", "gpu", "cpu"]}, "assigned_devices": {"main": "auto"}, "asked_device": "auto", "os": "Version 27.2.1 (Build 26B131)"}; first request 311.47425 ms

Omni Parse Bench, 2937 pages (headline = mean of the families):

| system | headline | text | tables | layout |
|---|---|---|---|---|
| text | 29.11 | 31.3 | 0.0 | 56.1 |
| docs | 39.77 | 25.2 | 20.3 | 73.9 |
| Indra OCR | 78.80 | 68.0 | 75.8 | 92.6 |
| Tesseract | 5.91 | 17.7 | 0.0 | 0.0 |
