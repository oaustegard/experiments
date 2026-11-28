# Apple Vision OCR on Omni Parse Bench: Apple M5 Max, 2026-11-27T17:41:39

macOS 27.2.1, AC True, low power False. 40 pages (0 could not be rasterised).

| arm | GPU | pages | ms/page median | mean | p90 | pages/s | idle median, same pages | errors | GPU tok/s alone -> during |
|---|---|---|---|---|---|---|---|---|---|
| text | idle | 40 | 184.8 | 214.1 | 327.6 | 4.67 |  | 0 |  |
| text | busy | 10 | 316.1 | 357.2 | 561.8 | 2.8 | 208.2 | 0 | 45.38 -> None (busy 1.0) |
| docs | idle | 40 | 289.2 | 654.6 | 463.7 | 1.53 |  | 0 |  |
| docs | busy | 10 | 422.0 | 444.3 | 710.6 | 2.25 | 353.3 | 0 | 45.38 -> None (busy 1.0) |

- text: {"request": "VNRecognizeTextRequest", "level": "accurate", "revision": 3, "language_correction": true, "automatically_detects_language": true, "asked_device": "auto", "supported_devices": {"VNComputeStageMain": ["MLNeuralEngineComputeDevice", "MLGPUComputeDevice", "MLCPUComputeDevice"]}, "assigned_devices": {"VNComputeStageMain": "auto"}}; first request 52779.23 ms
- docs: {"request": "RecognizeDocumentsRequest.revision1", "swiftc": "Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)\nTarget: arm64-apple-macosx27.2.0", "os": "Version 27.2.1 (Build 26B131)", "supported_devices": {"main": ["ne", "gpu", "cpu"]}, "asked_device": "auto", "assigned_devices": {"main": "auto"}}; first request 37426.194833 ms

Not graded here: run ocria-pack/run/grade.py on the arm folders (see README).
