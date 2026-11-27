// RecognizeDocumentsRequest (Vision, macOS 26) as a line-oriented CLI for ocr_ane.py.
//
// The request is a Swift struct with no Objective-C counterpart, so pyobjc cannot reach it.
// Reads one image path per line on stdin, writes one JSON object per line on stdout:
//   {"path", "width", "height", "ms", "documents": [{"title", "paragraphs", "tables", "lists"}]}
// Boxes are [x, y, width, height] in pixels, origin top-left. The first line written is a
// header with the request's supported compute devices and the device asked for.
//
//   swiftc -O -parse-as-library docs_ocr.swift -o docs_ocr
//   ls *.png | ./docs_ocr [--device auto|ne|gpu|cpu]
//
// Written from Apple's documentation without a compiler (2026-11-27); ocr_ane.py records
// the compile error and carries on without this arm if it does not build.

import CoreGraphics
import CoreML
import Foundation
import ImageIO
import Vision

func emit(_ obj: [String: Any]) {
    let data = (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{\"error\":\"unserializable\"}".utf8)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

func deviceName(_ d: MLComputeDevice) -> String {
    switch d {
    case .cpu: return "cpu"
    case .gpu: return "gpu"
    case .neuralEngine: return "ne"
    @unknown default: return "unknown"
    }
}

@available(macOS 26.0, *)
func box(_ r: NormalizedRegion, _ size: CGSize) -> [Double] {
    let b = r.boundingBox.toImageCoordinates(size, origin: .upperLeft)
    return [Double(b.origin.x), Double(b.origin.y), Double(b.size.width), Double(b.size.height)]
}

@available(macOS 26.0, *)
func describe(_ d: DocumentObservation.Container, _ size: CGSize) -> [String: Any] {
    var doc: [String: Any] = [:]
    if let t = d.title {
        doc["title"] = t.transcript
    }
    doc["paragraphs"] = d.paragraphs.map { p -> [String: Any] in
        let lines = p.lines.map { l -> [String: Any] in
            ["text": l.transcript, "bbox": box(l.boundingRegion, size), "title": l.isTitle]
        }
        return ["text": p.transcript, "bbox": box(p.boundingRegion, size), "lines": lines]
    }
    doc["tables"] = d.tables.map { t -> [String: Any] in
        var cells: [[String: Any]] = []
        for row in t.rows {
            for c in row {
                cells.append(["r0": c.rowRange.lowerBound, "r1": c.rowRange.upperBound,
                              "c0": c.columnRange.lowerBound, "c1": c.columnRange.upperBound,
                              "text": c.content.text.transcript])
            }
        }
        return ["bbox": box(t.boundingRegion, size), "rows": t.rows.count, "cols": t.columns.count,
                "cells": cells]
    }
    doc["lists"] = d.lists.map { l -> [String: Any] in
        let items = l.items.map { i -> [String: Any] in ["marker": i.markerString, "text": i.itemString] }
        return ["bbox": box(l.boundingRegion, size), "items": items]
    }
    return doc
}

@available(macOS 26.0, *)
func run(device: String) async {
    var request = RecognizeDocumentsRequest()
    var supported: [String: [String]] = [:]
    for (stage, devices) in request.supportedComputeStageDevices {
        supported["\(stage)"] = devices.map(deviceName)
        if device != "auto", let d = devices.first(where: { deviceName($0) == device }) {
            request.setComputeDevice(d, for: stage)
        }
    }
    var assigned: [String: String] = [:]
    for stage in request.supportedComputeStageDevices.keys {
        assigned["\(stage)"] = request.computeDevice(for: stage).map(deviceName) ?? "auto"
    }
    emit(["header": true, "request": "\(request)", "supported_devices": supported,
          "assigned_devices": assigned, "asked_device": device,
          "os": ProcessInfo.processInfo.operatingSystemVersionString])

    while let line = readLine() {
        let path = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if path.isEmpty { continue }
        var out: [String: Any] = ["path": path]
        let t0 = DispatchTime.now().uptimeNanoseconds
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            out["error"] = "unreadable image"
            emit(out)
            continue
        }
        let size = CGSize(width: img.width, height: img.height)
        out["width"] = img.width
        out["height"] = img.height
        let t1 = DispatchTime.now().uptimeNanoseconds
        do {
            let observations = try await request.perform(on: img)
            let t2 = DispatchTime.now().uptimeNanoseconds
            out["ms_decode"] = Double(t1 - t0) / 1e6
            out["ms"] = Double(t2 - t1) / 1e6
            out["documents"] = observations.map { o -> [String: Any] in
                var d = describe(o.document, size)
                d["confidence"] = Double(o.confidence)
                return d
            }
        } catch {
            out["error"] = "\(error)"
        }
        emit(out)
    }
}

@main
struct Main {
    static func main() async {
        var device = "auto"
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--device"), i + 1 < args.count {
            device = args[i + 1]
        }
        if #available(macOS 26.0, *) {
            await run(device: device)
        } else {
            emit(["header": true, "error": "RecognizeDocumentsRequest needs macOS 26"])
            exit(2)
        }
    }
}
