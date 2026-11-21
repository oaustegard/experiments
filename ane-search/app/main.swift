// Projects Search: menu-bar app with a global shortcut. Queries are embedded with the bundled Core ML model
// (leaf-ir, 32 tokens) and ranked in-process against the remax_kb knowledge base that index.py maintains. No server, no Python.
import AppKit
import SwiftUI
import CoreML
import Accelerate
import Carbon.HIToolbox
import ServiceManagement
import Combine

let defaults = UserDefaults.standard
let home = FileManager.default.homeDirectoryForCurrentUser.path
let indexDir = defaults.string(forKey: "indexDir") ?? home + "/Projects/ane-search/index"
let rootDir = defaults.string(forKey: "root") ?? home + "/Projects"
let reindexScript = (indexDir as NSString).deletingLastPathComponent + "/reindex.sh"
let queryPrefix = "Represent this sentence for searching relevant passages: "
let S = 32, DIM = 768

// MARK: tokenizer (BERT uncased WordPiece, matching the tokenizers library for this model)
struct WordPiece {
    let vocab: [String: Int32]
    init(url: URL) throws {
        var v = [String: Int32]()
        let text = try String(contentsOf: url, encoding: .utf8)
        var i: Int32 = 0
        text.enumerateLines { line, _ in v[line] = i; i += 1 }
        vocab = v
    }
    static func isPunct(_ u: Unicode.Scalar) -> Bool {
        let c = u.value
        if (33...47).contains(c) || (58...64).contains(c) || (91...96).contains(c) || (123...126).contains(c) { return true }
        switch u.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation, .initialPunctuation, .finalPunctuation, .otherPunctuation: return true
        default: return false
        }
    }
    static func isCJK(_ c: UInt32) -> Bool {
        (0x4E00...0x9FFF).contains(c) || (0x3400...0x4DBF).contains(c) || (0x20000...0x2A6DF).contains(c) || (0x2A700...0x2B73F).contains(c) ||
        (0x2B740...0x2B81F).contains(c) || (0x2B820...0x2CEAF).contains(c) || (0xF900...0xFAFF).contains(c) || (0x2F800...0x2FA1F).contains(c)
    }
    func words(_ text: String) -> [[Unicode.Scalar]] {
        var out = [[Unicode.Scalar]](), cur = [Unicode.Scalar]()
        func flush() { if !cur.isEmpty { out.append(cur); cur = [] } }
        for u in text.lowercased().decomposedStringWithCanonicalMapping.unicodeScalars {
            let cat = u.properties.generalCategory
            if u.value == 0 || u.value == 0xFFFD { continue }
            if cat == .nonspacingMark { continue }                                   // strip accents
            if u == " " || u == "\t" || u == "\n" || u == "\r" || cat == .spaceSeparator { flush(); continue }
            if cat == .control || cat == .format { continue }
            if WordPiece.isPunct(u) || WordPiece.isCJK(u.value) { flush(); out.append([u]); continue }
            cur.append(u)
        }
        flush()
        return out
    }
    func encode(_ text: String, maxLen: Int) -> [Int32] {
        let unk = vocab["[UNK]"] ?? 100
        var ids: [Int32] = [vocab["[CLS]"] ?? 101]
        for w in words(text) {
            if w.count > 100 { ids.append(unk); continue }
            var start = 0, pieces = [Int32](), bad = false
            while start < w.count {
                var end = w.count, found: Int32? = nil
                while start < end {
                    var s = String(String.UnicodeScalarView(w[start..<end]))
                    if start > 0 { s = "##" + s }
                    if let id = vocab[s] { found = id; break }
                    end -= 1
                }
                guard let id = found else { bad = true; break }
                pieces.append(id); start = end
            }
            ids.append(contentsOf: bad ? [unk] : pieces)
        }
        if ids.count > maxLen - 1 { ids = Array(ids[0..<(maxLen - 1)]) }
        ids.append(vocab["[SEP]"] ?? 102)
        return ids
    }
}

// MARK: embedder
final class Embedder {
    let model: MLModel, tok: WordPiece
    init() throws {
        let res = Bundle.main.resourceURL!
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuOnly            // at 32 tokens the CPU ties the Neural Engine and loads faster
        model = try MLModel(contentsOf: res.appendingPathComponent("leaf_S32.mlmodelc"), configuration: cfg)
        tok = try WordPiece(url: res.appendingPathComponent("vocab.txt"))
    }
    func embed(_ query: String) throws -> [Float] {
        let ids = tok.encode(queryPrefix + query, maxLen: S)
        let x = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .int32)
        let m = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .float16)
        for i in 0..<S { x[i] = NSNumber(value: i < ids.count ? ids[i] : 0); m[i] = NSNumber(value: i < ids.count ? 1.0 : 0.0) }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": x, "pad_mask": m]))
        guard let e = out.featureValue(for: "embedding")?.multiArrayValue else { throw NSError(domain: "embed", code: 1) }
        var v = (0..<DIM).map { e[$0].floatValue }
        var n: Float = 0; vDSP_svesq(v, 1, &n, vDSP_Length(DIM)); n = sqrt(n)
        if n > 0 { vDSP_vsdiv(v, 1, &n, &v, 1, vDSP_Length(DIM)) }
        return v
    }
}

// MARK: index (remax_kb v2 knowledge base: projects.kbi + projects.kbc/, remex 4-bit codes; written by index.py via kb.py)
struct Passage { let path: String; let head: String; let line: Int; let text: String }
struct Hit: Identifiable { let id: Int; let p: Passage; let cos: Float; let bm25: Float }
func fail(_ s: String) -> NSError { NSError(domain: "index", code: 2, userInfo: [NSLocalizedDescriptionKey: s]) }

extension Data {
    func le<T: FixedWidthInteger>(_ o: Int, _: T.Type) -> Int { withUnsafeBytes { Int(T(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: T.self))) } }
    func array<T>(_ r: Range<Int>, _: T.Type) -> [T] {            // little-endian host; copies, so alignment inside the zip does not matter
        let n = r.count / MemoryLayout<T>.stride
        return [T](unsafeUninitializedCapacity: n) { buf, count in
            withUnsafeBytes { src in UnsafeMutableRawBufferPointer(buf).copyMemory(from: UnsafeRawBufferPointer(rebasing: src[r.lowerBound..<r.lowerBound + n * MemoryLayout<T>.stride])) }
            count = n
        }
    }
}

/// A zip whose entries are all stored uncompressed, which is how remax_kb writes a .kbi: entries are byte ranges of the mapped file.
struct StoredZip {
    let data: Data
    var entries = [String: Range<Int>]()
    init(_ path: String) throws {
        data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
        var e = data.count - 22
        while e >= 0 && data.le(e, UInt32.self) != 0x06054b50 { e -= 1 }
        guard e >= 0 else { throw fail("\(path) is not a zip") }
        var p = data.le(e + 16, UInt32.self)
        for _ in 0..<data.le(e + 10, UInt16.self) {
            guard p + 46 <= data.count, data.le(p, UInt32.self) == 0x02014b50 else { throw fail("bad zip directory") }
            let nameLen = data.le(p + 28, UInt16.self), size = data.le(p + 20, UInt32.self), local = data.le(p + 42, UInt32.self)
            guard data.le(p + 10, UInt16.self) == 0 else { throw fail("compressed zip entry; a .kbi stores entries uncompressed") }
            let name = String(decoding: data.subdata(in: p + 46..<p + 46 + nameLen), as: UTF8.self)
            let start = local + 30 + data.le(local + 26, UInt16.self) + data.le(local + 28, UInt16.self)
            guard start + size <= data.count else { throw fail("zip entry \(name) runs past the file") }
            entries[name] = start..<start + size
            p += 46 + nameLen + data.le(p + 30, UInt16.self) + data.le(p + 32, UInt16.self)
        }
    }
    func range(_ name: String) throws -> Range<Int> { guard let r = entries[name] else { throw fail("the index has no \(name)") }; return r }
    func npyHeader(_ name: String) throws -> String { let r = try range(name), a = try npy(name); return String(decoding: data.subdata(in: r.lowerBound..<a.lowerBound), as: UTF8.self) }
    func npy(_ name: String) throws -> Range<Int> {                  // the array bytes of a .npy entry
        let r = try range(name), major = Int(data[r.lowerBound + 6])
        let hlen = major == 1 ? data.le(r.lowerBound + 8, UInt16.self) : data.le(r.lowerBound + 8, UInt32.self)
        return r.lowerBound + (major == 1 ? 10 : 12) + hlen..<r.upperBound
    }
}

final class Index {
    static let kbi = indexDir + "/projects.kbi"
    let zip: StoredZip
    var n = 0, live = 0, dim = 0                                     // rows (including tombstoned ones), live rows, coded dimensions
    var codes = 0..<0, map = 0..<0                                   // byte ranges in the .kbi: n x dim/2 code bytes, n x 24 chunk-map bytes
    var dead = [Bool](), rot = [Float](), cent = [Float]()
    var chunkDir = "", shards = [Int: Data]()
    var vocab = [String: Int32](), indptr = [Int](), docs = [Int32](), scores = [Float](), rowOfDoc = [Int32]()
    var hasKeywords: Bool { !vocab.isEmpty }
    var built = "", files = 0, stamp = Date.distantPast
    static func stampNow() -> Date { ((try? FileManager.default.attributesOfItem(atPath: indexDir + "/stats.json"))?[.modificationDate] as? Date) ?? .distantPast }

    init() throws {
        stamp = Index.stampNow()
        zip = try StoredZip(Index.kbi)
        guard let m = try JSONSerialization.jsonObject(with: zip.data.subdata(in: try zip.range("manifest.json"))) as? [String: Any],
              let b = m["binarizer"] as? [String: Any], let c = m["chunks"] as? [String: Any] else { throw fail("unreadable manifest") }
        guard m["spec_version"] as? String == "2", b["kind"] as? String == "remex-lloyd-max", b["bits"] as? Int == 4 else {
            throw fail("this app reads remax_kb v2 files with the remex 4-bit codec; got \(b["kind"] ?? "?") \(b["bits"] ?? "?")-bit")
        }
        dim = b["dim"] as? Int ?? 0; n = c["total_rows"] as? Int ?? 0
        codes = try zip.range("vectors.bin"); map = try zip.range("chunk_map.bin")
        guard dim > 0, dim % 2 == 0, dim <= DIM, codes.count == n * dim / 2, map.count == n * 24 else { throw fail("index sizes do not match its manifest") }
        rot = zip.data.array(try zip.range("remex/rotation.f32"), Float.self); cent = zip.data.array(try zip.range("remex/centroids.f32"), Float.self)
        guard rot.count == dim * dim, cent.count == 16 else { throw fail("rotation or centroid table has the wrong size") }
        dead = (0..<n).map { zip.data[map.lowerBound + $0 * 24 + 2] & 1 != 0 }
        rowOfDoc = (0..<n).filter { !dead[$0] }.map(Int32.init); live = rowOfDoc.count
        chunkDir = indexDir + "/" + (c["uri"] as? String ?? "projects.kbc/")
        if let s = try? JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: indexDir + "/stats.json"))) as? [String: Any] {
            built = s["built"] as? String ?? ""; files = s["files"] as? Int ?? 0
        }
    }
    /// bm25s tables: one column per token, holding the precomputed score of every document that contains it.
    func loadKeywords() {
        guard let v = try? zip.range("bm25/vocab.index.json"), let j = try? JSONSerialization.jsonObject(with: zip.data.subdata(in: v)) as? [String: NSNumber],
              let a = try? zip.npy("bm25/indptr.csc.index.npy"), let b = try? zip.npy("bm25/indices.csc.index.npy"), let c = try? zip.npy("bm25/data.csc.index.npy") else { return }
        let wide = (try? zip.npyHeader("bm25/indptr.csc.index.npy").contains("i8")) ?? false          // bm25s writes int64 offsets
        let ip = wide ? zip.data.array(a, Int64.self).map(Int.init) : zip.data.array(a, Int32.self).map(Int.init)
        let ix = zip.data.array(b, Int32.self), sc = zip.data.array(c, Float.self)
        guard ix.count == sc.count, !(ip.contains { $0 < 0 || $0 > ix.count }), !(ix.contains { $0 < 0 || Int($0) >= live }) else { return }
        indptr = ip; docs = ix; scores = sc; vocab = j.mapValues { $0.int32Value }
    }
    /// bm25s' default tokenizer: lowercase, then runs of two or more word characters (so response_model is one token).
    static let wordRe = try! NSRegularExpression(pattern: "\\b\\w\\w+\\b")
    static func words(_ t: String) -> [String] {
        let s = t.lowercased() as NSString
        return wordRe.matches(in: s as String, range: NSRange(location: 0, length: s.length)).map { s.substring(with: $0.range) }
    }
    /// Cosine of the query against every decoded passage, without decoding: rotate the query once, then each code byte (two 4-bit centroid numbers) is one table lookup.
    func semantic(_ qv: [Float]) -> [Float] {
        var q = Array(qv[0..<dim]), norm: Float = 0
        vDSP_svesq(q, 1, &norm, vDSP_Length(dim)); norm = sqrt(norm); if norm > 0 { vDSP_vsdiv(q, 1, &norm, &q, 1, vDSP_Length(dim)) }
        var qr = [Float](repeating: 0, count: dim)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(dim), Int32(dim), 1, rot, Int32(dim), q, 1, 0, &qr, 1)
        let half = dim / 2
        var table = [Float](repeating: 0, count: half * 256)
        for j in 0..<half { for b in 0..<256 { table[j << 8 | b] = cent[b >> 4] * qr[2 * j] + cent[b & 15] * qr[2 * j + 1] } }
        var sem = [Float](repeating: 0, count: n)
        zip.data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self) + codes.lowerBound
            table.withUnsafeBufferPointer { t in
                for r in 0..<n {
                    let row = base + r * half; var s: Float = 0
                    for j in 0..<half { s += t[j << 8 | Int(row[j])] }
                    sem[r] = s
                }
            }
        }
        return sem
    }
    func passage(_ row: Int) -> Passage? {
        let o = map.lowerBound + row * 24
        let sid = zip.data.le(o, UInt16.self), off = zip.data.le(o + 4, UInt64.self), len = zip.data.le(o + 12, UInt32.self)
        if shards[sid] == nil { shards[sid] = try? Data(contentsOf: URL(fileURLWithPath: chunkDir + String(format: "shard-%04d.bin", sid)), options: .mappedIfSafe) }
        guard let sh = shards[sid], off + len <= sh.count else { return nil }
        let rec = sh.subdata(in: off..<off + len)
        guard let nl = rec.firstIndex(of: 0x0A), let h = try? JSONSerialization.jsonObject(with: rec.subdata(in: 0..<nl)) as? [String: Any], let m = h["meta"] as? [String: Any] else { return nil }
        var text = String(decoding: rec.subdata(in: nl + 1..<rec.count), as: UTF8.self)       // first line repeats "path — heading"
        if let t = text.firstIndex(of: "\n") { text = String(text[text.index(after: t)...]) }
        return Passage(path: m["path"] as? String ?? "", head: m["head"] as? String ?? "", line: m["line"] as? Int ?? 1, text: text)
    }
    func search(_ q: String, _ qv: [Float], k: Int = 20) -> [Hit] {
        let sem = semantic(qv)
        var lex = [Float](repeating: 0, count: n)
        for w in Index.words(q) {
            guard let t = vocab[w].map(Int.init), t + 1 < indptr.count, indptr[t] <= indptr[t + 1] else { continue }
            for i in indptr[t]..<indptr[t + 1] { lex[Int(rowOfDoc[Int(docs[i])])] += scores[i] }
        }
        let cand = (0..<n).filter { !dead[$0] }
        let depth = 200
        let rs = cand.sorted { sem[$0] != sem[$1] ? sem[$0] > sem[$1] : $0 < $1 }.prefix(depth)
        let rl = cand.filter { lex[$0] > 0 }.sorted { lex[$0] != lex[$1] ? lex[$0] > lex[$1] : $0 < $1 }.prefix(depth)
        var score = [Int: Float]()
        for (r, i) in rs.enumerated() { score[i, default: 0] += 1 / Float(60 + r) }
        for (r, i) in rl.enumerated() { score[i, default: 0] += 1 / Float(60 + r) }
        var perFile = [String: Int](), out = [Hit]()
        for i in score.keys.sorted(by: { score[$0]! != score[$1]! ? score[$0]! > score[$1]! : $0 < $1 }) {
            guard let p = passage(i) else { continue }
            if perFile[p.path, default: 0] >= 2 { continue }
            perFile[p.path, default: 0] += 1
            out.append(Hit(id: i, p: p, cos: sem[i], bm25: lex[i]))
            if out.count >= k { break }
        }
        return out
    }
}

// MARK: engine (loads in the background, reloads when index.py has written a newer index)
final class Engine {
    private(set) var index: Index?
    private var embedder: Embedder?
    private let q = DispatchQueue(label: "engine", qos: .userInitiated)
    var onReady: ((String) -> Void)?
    func load() {
        q.async {
            do {
                if self.embedder == nil { self.embedder = try Embedder() }
                let ix = try Index(); self.index = ix
                self.report()
                ix.loadKeywords(); self.report()
            } catch { DispatchQueue.main.async { self.onReady?("Could not load: \(error.localizedDescription)") } }
        }
    }
    private func report() {
        guard let ix = index else { return }
        let s = "\(ix.live.formatted()) passages from \(ix.files.formatted()) files · indexed \(ix.built)" + (ix.hasKeywords ? "" : " · loading keywords…")
        DispatchQueue.main.async { self.onReady?(s) }
    }
    func refreshIfStale() { if let ix = index, Index.stampNow() != ix.stamp { load() } }
    func search(_ text: String, done: @escaping ([Hit], Double, Double) -> Void) {
        q.async {
            guard let ix = self.index, let e = self.embedder else { return }
            let t0 = CFAbsoluteTimeGetCurrent()
            guard let qv = try? e.embed(text) else { return }
            let t1 = CFAbsoluteTimeGetCurrent()
            let hits = ix.search(text, qv)
            let t2 = CFAbsoluteTimeGetCurrent()
            DispatchQueue.main.async { done(hits, (t1 - t0) * 1000, (t2 - t1) * 1000) }
        }
    }
    func searchSync(_ text: String) throws -> ([Hit], Double, Double) {
        let e = try Embedder(); let ix = try Index(); ix.loadKeywords()
        _ = try e.embed("warm up")
        let t0 = CFAbsoluteTimeGetCurrent(); let qv = try e.embed(text); let t1 = CFAbsoluteTimeGetCurrent()
        let h = ix.search(text, qv); let t2 = CFAbsoluteTimeGetCurrent()
        return (h, (t1 - t0) * 1000, (t2 - t1) * 1000)
    }
}

// MARK: UI
final class Model: ObservableObject {
    @Published var query = "" { didSet { if query != oldValue { schedule() } } }
    @Published var hits = [Hit]()
    @Published var selected = 0
    @Published var status = "Loading…"
    @Published var focusTick = 0
    let engine = Engine()
    private var seq = 0, idle = ""
    init() {
        engine.onReady = { [weak self] s in self?.idle = s; if self?.query.isEmpty ?? true { self?.status = s } else { self?.schedule() } }
        engine.load()
    }
    func schedule() {
        seq += 1; let me = seq, text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { hits = []; status = idle; return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.04) {
            guard me == self.seq else { return }
            self.engine.search(text) { h, e, r in
                guard me == self.seq else { return }
                self.hits = h; self.selected = 0
                self.status = String(format: "%d results · embedded in %.1f ms, ranked in %.1f ms", h.count, e, r)
            }
        }
    }
    func move(_ d: Int) { if !hits.isEmpty { selected = min(max(selected + d, 0), hits.count - 1) } }
    func open(_ i: Int? = nil, reveal: Bool = false) {
        let i = i ?? selected; guard hits.indices.contains(i) else { return }
        let url = URL(fileURLWithPath: rootDir + "/" + hits[i].p.path)
        if reveal { NSWorkspace.shared.activateFileViewerSelecting([url]) } else { NSWorkspace.shared.open(url) }
        (NSApp.delegate as? AppDelegate)?.hidePanel()
    }
}

struct SearchView: View {
    @ObservedObject var m: Model
    @FocusState private var focused: Bool
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "text.magnifyingglass").font(.system(size: 20)).foregroundStyle(.secondary)
                TextField("Search your write-ups by meaning…", text: $m.query)
                    .textFieldStyle(.plain).font(.system(size: 22)).focused($focused)
                    .onSubmit { m.open() }
            }.padding(.horizontal, 18).padding(.vertical, 14)
            if !m.hits.isEmpty {
                Divider()
                ScrollViewReader { sp in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(Array(m.hits.enumerated()), id: \.element.id) { i, h in
                                Row(h: h, on: i == m.selected).id(i)
                                    .contentShape(Rectangle())
                                    .onTapGesture { m.open(i, reveal: NSEvent.modifierFlags.contains(.command)) }
                            }
                        }.padding(6)
                    }.frame(maxHeight: 440)
                    .onChange(of: m.selected) { _, s in sp.scrollTo(s) }
                }
            }
            Divider()
            HStack {
                Text(m.status).lineLimit(1)
                Spacer()
                Text("↩ open   ⌘↩ show in Finder   esc close")
            }.font(.system(size: 11)).foregroundStyle(.secondary).padding(.horizontal, 14).padding(.vertical, 7)
        }
        .frame(width: 720)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .onChange(of: m.focusTick) { _, _ in focused = true }
        .onAppear { focused = true }
    }
}

struct Row: View {
    let h: Hit, on: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(h.p.path):\(h.p.line)").font(.system(size: 11, design: .monospaced)).foregroundStyle(on ? Color.white.opacity(0.85) : Color.accentColor).lineLimit(1).truncationMode(.middle)
            if !h.p.head.isEmpty { Text(h.p.head).font(.system(size: 13, weight: .semibold)).lineLimit(1) }
            Text(h.p.text.prefix(260).replacingOccurrences(of: "\n", with: " ")).font(.system(size: 12)).foregroundStyle(on ? Color.white.opacity(0.9) : Color.secondary).lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 10).padding(.vertical, 7)
        .foregroundStyle(on ? Color.white : Color.primary)
        .background(RoundedRectangle(cornerRadius: 8).fill(on ? Color.accentColor : Color.clear))
    }
}

final class Panel: NSPanel {
    override var canBecomeKey: Bool { true }
    override func cancelOperation(_ sender: Any?) { (NSApp.delegate as? AppDelegate)?.hidePanel() }
}

// MARK: app
var hotKeyHandler: (() -> Void)?
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    let model = Model()
    var item: NSStatusItem!, panel: Panel!, monitor: Any?, hotKey: EventHotKeyRef?
    var loginItem: NSMenuItem!
    var hotKeyStatus: OSStatus = -1
    var host: NSHostingView<SearchView>!, sub: AnyCancellable?

    func applicationDidFinishLaunching(_ n: Notification) {
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = NSImage(systemSymbolName: "text.magnifyingglass", accessibilityDescription: "Projects Search")
        let menu = NSMenu()
        menu.addItem(withTitle: "Search…", action: #selector(toggle), keyEquivalent: " ").keyEquivalentModifierMask = [.control, .option]
        menu.addItem(withTitle: "Re-index now", action: #selector(reindex), keyEquivalent: "")
        menu.addItem(.separator())
        loginItem = menu.addItem(withTitle: "Open at login", action: #selector(toggleLogin), keyEquivalent: "")
        loginItem.state = SMAppService.mainApp.status == .enabled ? .on : .off
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Projects Search", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        item.menu = menu

        panel = Panel(contentRect: NSRect(x: 0, y: 0, width: 720, height: 120), styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView], backing: .buffered, defer: false)
        panel.isFloatingPanel = true; panel.level = .floating; panel.backgroundColor = .clear; panel.isOpaque = false; panel.hasShadow = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]; panel.hidesOnDeactivate = false; panel.delegate = self
        host = NSHostingView(rootView: SearchView(m: model)); host.sizingOptions = [.intrinsicContentSize]
        panel.contentView = host
        sub = model.$hits.sink { [weak self] _ in DispatchQueue.main.async { self?.fit() } }     // grow and shrink with the result list

        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] e in
            guard let self, self.panel.isKeyWindow else { return e }
            switch Int(e.keyCode) {
            case kVK_DownArrow: self.model.move(1); return nil
            case kVK_UpArrow: self.model.move(-1); return nil
            case kVK_Escape: self.hidePanel(); return nil
            case kVK_Return where e.modifierFlags.contains(.command): self.model.open(reveal: true); return nil
            default: return e
            }
        }
        registerHotKey()
        // a click anywhere outside the panel closes it, also when the panel never became the key window
        NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in if self?.panel.isVisible == true { self?.hidePanel() } }
        if let i = CommandLine.arguments.firstIndex(of: "--selftest"), CommandLine.arguments.count > i + 2 { selfTest(CommandLine.arguments[i + 1], CommandLine.arguments[i + 2]) }
        if CommandLine.arguments.contains("--show") { DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.showPanel() } }
    }
    /// Opening the app again (Spotlight, Finder, `open`) shows the panel, so the shortcut is not the only way in.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { if !panel.isVisible { toggle() }; return false }
    func registerHotKey() {
        hotKeyHandler = { [weak self] in self?.toggle() }
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, _, _ in DispatchQueue.main.async { hotKeyHandler?() }; return noErr }, 1, &spec, nil, nil)
        let key = defaults.object(forKey: "hotKeyCode") as? Int ?? kVK_Space
        let mods = defaults.object(forKey: "hotKeyModifiers") as? Int ?? (controlKey | optionKey)
        let st = RegisterEventHotKey(UInt32(key), UInt32(mods), EventHotKeyID(signature: OSType(0x50535243), id: 1), GetApplicationEventTarget(), 0, &hotKey)
        hotKeyStatus = st
        if st != noErr { NSLog("Projects Search: could not register the shortcut (status \(st)); another app may hold it") }
    }
    /// Headless check of the real UI: show the panel, type a query through the view model, move the selection, save a picture of the panel.
    func selfTest(_ query: String, _ out: String) {
        func wait(_ s: Double, _ f: @escaping () -> Void) { DispatchQueue.main.asyncAfter(deadline: .now() + s, execute: f) }
        wait(2.5) {
            self.toggle()                                              // the same entry point the shortcut calls
            wait(0.4) {
                self.model.query = query
                wait(1.2) {
                    self.model.move(1)
                    wait(0.5) {
                        let v = self.panel.contentView!
                        if let rep = v.bitmapImageRepForCachingDisplay(in: v.bounds) { v.cacheDisplay(in: v.bounds, to: rep); try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: out + ".png")) }
                        let info: [String: Any] = ["hotkey_registered": self.hotKeyStatus == noErr, "hotkey_status": Int(self.hotKeyStatus), "panel_visible": self.panel.isVisible,
                                                   "panel_key": self.panel.isKeyWindow, "text_field_focused": self.panel.firstResponder is NSTextView,
                                                   "hits": self.model.hits.count, "selected": self.model.selected, "status": self.model.status,
                                                   "first": self.model.hits.first?.p.path ?? "", "frame": NSStringFromRect(self.panel.frame), "keywords_loaded": (self.model.engine.index?.hasKeywords ?? false)]
                        try? JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: out + ".json"))
                        self.toggle()
                        wait(0.3) { NSApp.terminate(nil) }
                    }
                }
            }
        }
    }
    /// Size the panel to its content, keeping the top edge where it is.
    func fit() {
        host.invalidateIntrinsicContentSize(); host.layoutSubtreeIfNeeded()
        let h = host.intrinsicContentSize.height, f = panel.frame
        if h > 40, abs(h - f.height) > 0.5 { panel.setFrame(NSRect(x: f.minX, y: f.maxY - h, width: 720, height: h), display: true) }
    }
    @objc func toggle() {
        panel.isVisible ? hidePanel() : showPanel()
        if let i = CommandLine.arguments.firstIndex(of: "--debuglog"), CommandLine.arguments.count > i + 1 {
            let path = CommandLine.arguments[i + 1]
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                let line = "visible=\(self.panel.isVisible) key=\(self.panel.isKeyWindow) active=\(NSApp.isActive) field=\(self.panel.firstResponder is NSTextView)\n"
                if let h = FileHandle(forWritingAtPath: path) { h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); h.closeFile() } else { try? line.write(toFile: path, atomically: true, encoding: .utf8) }
            }
        }
    }
    func showPanel() {
        model.engine.refreshIfStale()
        let screen = NSScreen.screens.first { NSMouseInRect(NSEvent.mouseLocation, $0.frame, false) } ?? NSScreen.main
        if let f = screen?.visibleFrame {
            fit()
            panel.setFrameTopLeftPoint(NSPoint(x: f.midX - 360, y: f.maxY - f.height * 0.18))
        }
        NSApp.activate(ignoringOtherApps: true)
        NSRunningApplication.current.activate(options: [.activateAllWindows])
        panel.orderFrontRegardless()
        panel.makeKeyAndOrderFront(nil)
        model.focusTick += 1
        DispatchQueue.main.async { if let tv = self.panel.firstResponder as? NSTextView { tv.selectAll(nil) } }
    }
    func hidePanel() { panel.orderOut(nil) }
    func windowDidResignKey(_ n: Notification) { hidePanel() }
    @objc func reindex() {
        model.status = "Re-indexing…"
        DispatchQueue.global().async {
            let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/sh"); p.arguments = [reindexScript]
            try? p.run(); p.waitUntilExit()
            DispatchQueue.main.async { self.model.engine.refreshIfStale() }
        }
    }
    @objc func toggleLogin() {
        do {
            if SMAppService.mainApp.status == .enabled { try SMAppService.mainApp.unregister() } else { try SMAppService.mainApp.register() }
        } catch { NSLog("Projects Search: login item: \(error.localizedDescription)") }
        loginItem.state = SMAppService.mainApp.status == .enabled ? .on : .off
    }
}

// MARK: entry (headless modes for testing: --tokenize "text", --search "query")
let args = CommandLine.arguments
if let i = args.firstIndex(of: "--tokenize"), args.count > i + 1 {
    let tok = try WordPiece(url: Bundle.main.resourceURL!.appendingPathComponent("vocab.txt"))
    print(tok.encode(args[i + 1], maxLen: 512).map(String.init).joined(separator: " "))
    exit(0)
}
if let i = args.firstIndex(of: "--search"), args.count > i + 1 {
    let (hits, e, r) = try Engine().searchSync(args[i + 1])
    let rows = hits.map { ["i": $0.id, "path": $0.p.path, "line": $0.p.line, "cos": Double(($0.cos * 1000).rounded() / 1000), "bm25": Double(($0.bm25 * 10).rounded() / 10)] as [String: Any] }
    let out: [String: Any] = ["embed_ms": (e * 100).rounded() / 100, "rank_ms": (r * 100).rounded() / 100, "results": rows]
    print(String(data: try JSONSerialization.data(withJSONObject: out), encoding: .utf8)!)
    exit(0)
}
if let i = args.firstIndex(of: "--batch"), args.count > i + 1 {                 // one query per line in, one JSON line out (for the parity check)
    let e = try Embedder(); let ix = try Index(); ix.loadKeywords()
    for q in try String(contentsOfFile: args[i + 1], encoding: .utf8).split(separator: "\n").map(String.init) {
        let t0 = CFAbsoluteTimeGetCurrent(); let qv = try e.embed(q); let t1 = CFAbsoluteTimeGetCurrent(); let h = ix.search(q, qv); let t2 = CFAbsoluteTimeGetCurrent()
        let o: [String: Any] = ["q": q, "ids": h.map { $0.id }, "cos": h.map { Double($0.cos) }, "bm25": h.map { Double($0.bm25) }, "embed_ms": (t1 - t0) * 1000, "rank_ms": (t2 - t1) * 1000]
        print(String(data: try JSONSerialization.data(withJSONObject: o), encoding: .utf8)!)
    }
    exit(0)
}
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
