//! Loading SVG files into a `usvg::Tree`, plus the synthetic documents used
//! for the empty and error states.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use resvg::usvg;

pub struct Document {
    /// `None` for the synthetic welcome/error documents.
    pub path: Option<PathBuf>,
    pub tree: usvg::Tree,
    mtime: Option<SystemTime>,
}

impl Document {
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let tree = parse(&data, Some(path))?;
        Ok(Self {
            path: Some(path.to_path_buf()),
            tree,
            mtime: mtime_of(path),
        })
    }

    /// A document built from an SVG string we generated ourselves.
    pub fn synthetic(svg: &str) -> Self {
        let tree = parse(svg.as_bytes(), None)
            // The synthetic sources are compile-time constants shaped by
            // `escape`, so a parse failure here is a bug, not bad input.
            .expect("synthetic SVG failed to parse");
        Self {
            path: None,
            tree,
            mtime: None,
        }
    }

    pub fn size(&self) -> (f32, f32) {
        let s = self.tree.size();
        (s.width(), s.height())
    }

    pub fn title(&self) -> String {
        match &self.path {
            Some(p) => p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string()),
            None => "svgview".to_string(),
        }
    }

    /// True when the file backing this document has been written since we
    /// last read it. Drives the editor-friendly auto-reload.
    pub fn changed_on_disk(&self) -> bool {
        let Some(path) = &self.path else {
            return false;
        };
        match (mtime_of(path), self.mtime) {
            (Some(now), Some(then)) => now != then,
            // File appeared or vanished under us — treat as a change so the
            // reload path gets a chance to report the real error.
            (a, b) => a.is_some() != b.is_some(),
        }
    }

    /// Re-reads from disk. On failure the old tree is kept and the error is
    /// returned, so a half-saved file mid-edit does not blank the window.
    pub fn reload(&mut self) -> Result<(), String> {
        let Some(path) = self.path.clone() else {
            return Ok(());
        };
        let fresh = Self::load(&path)?;
        self.tree = fresh.tree;
        self.mtime = fresh.mtime;
        Ok(())
    }
}

fn mtime_of(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

fn parse(data: &[u8], path: Option<&Path>) -> Result<usvg::Tree, String> {
    let mut opt = usvg::Options {
        resources_dir: path.and_then(|p| p.parent()).map(|d| d.to_path_buf()),
        ..usvg::Options::default()
    };

    // Enumerating system fonts costs real milliseconds and most SVGs in the
    // wild — icons, logos, plots with paths — have no text at all.
    if may_contain_text(data) {
        opt.fontdb_mut().load_system_fonts();
    }

    usvg::Tree::from_data(data, &opt).map_err(|e| e.to_string())
}

/// Cheap conservative pre-scan. False positives only cost a font-db load.
fn may_contain_text(data: &[u8]) -> bool {
    // SVGZ: we would have to inflate to look, and usvg is about to do that
    // anyway. Just assume text is present.
    if data.starts_with(&[0x1f, 0x8b]) {
        return true;
    }
    const NEEDLES: [&[u8]; 3] = [b"<text", b"<tspan", b"font-family"];
    NEEDLES
        .iter()
        .any(|n| data.windows(n.len()).any(|w| w == *n))
}

const FG: &str = "#c8c8c8";
const ACCENT: &str = "#6fa8dc";
const PANEL: &str = "#1e1e1e";

/// Wraps generated chrome in an opaque panel. Without it these documents would
/// be composited over whichever background the user last cycled to, and the
/// light-on-dark text would vanish against `Background::White`.
fn chrome_svg(width: i32, height: i32, body: &str) -> String {
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="{PANEL}"/>
  {body}
</svg>"##
    )
}

pub fn welcome_svg() -> String {
    chrome_svg(
        520,
        200,
        &format!(
            r##"<path d="M45 150 C 75 55, 145 55, 175 150" fill="none" stroke="{ACCENT}" stroke-width="7" stroke-linecap="round"/>
  <g stroke="{ACCENT}" stroke-width="2" opacity="0.55">
    <line x1="45" y1="150" x2="75" y2="55"/>
    <line x1="175" y1="150" x2="145" y2="55"/>
  </g>
  <g fill="{ACCENT}">
    <rect x="37" y="142" width="16" height="16"/>
    <rect x="167" y="142" width="16" height="16"/>
  </g>
  <g fill="none" stroke="{ACCENT}" stroke-width="2.5">
    <circle cx="75" cy="55" r="6"/>
    <circle cx="145" cy="55" r="6"/>
  </g>
  <text x="250" y="82" font-family="Segoe UI, Arial, sans-serif" font-size="26" fill="{FG}">svgview</text>
  <text x="250" y="114" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="{FG}" opacity="0.75">Drop an SVG here, or press Ctrl+O.</text>
  <text x="250" y="140" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="{FG}" opacity="0.75">Press H for the key list.</text>"##
        ),
    )
}

pub fn message_svg(heading: &str, body: &str) -> String {
    let lines = wrap(body, 58);
    let height = 96 + 24 * lines.len().max(1) as i32;
    let mut text = String::new();
    for (i, line) in lines.iter().enumerate() {
        text.push_str(&format!(
            r#"<text x="40" y="{y}" font-family="Consolas, Menlo, monospace" font-size="15" fill="{FG}" opacity="0.85">{line}</text>"#,
            y = 100 + 24 * i as i32,
            line = escape(line),
        ));
    }
    chrome_svg(
        640,
        height,
        &format!(
            r##"<text x="40" y="58" font-family="Segoe UI, Arial, sans-serif" font-size="22" fill="#e06c75">{heading}</text>
  {text}"##,
            heading = escape(heading),
        ),
    )
}

pub fn help_svg() -> String {
    const KEYS: &[(&str, &str)] = &[
        ("Ctrl+O", "open a file"),
        ("drag &amp; drop", "open a dropped file"),
        ("wheel", "zoom at the pointer"),
        ("drag / arrows", "pan"),
        ("Ctrl+0", "fit to window"),
        ("Ctrl+1", "actual size (100%)"),
        ("Ctrl++ / Ctrl+-", "zoom in / out"),
        ("B", "cycle background"),
        ("R", "reload from disk"),
        ("Ctrl+S", "export PNG at current zoom"),
        ("F11", "fullscreen"),
        ("H", "toggle this help"),
        ("Esc / Ctrl+W", "close"),
    ];
    let height = 80 + 26 * KEYS.len() as i32;
    let mut rows = String::new();
    for (i, (key, what)) in KEYS.iter().enumerate() {
        let y = 76 + 26 * i as i32;
        rows.push_str(&format!(
            r#"<text x="40" y="{y}" font-family="Consolas, Menlo, monospace" font-size="15" fill="{ACCENT}">{key}</text>
  <text x="220" y="{y}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="{FG}">{what}</text>"#
        ));
    }
    chrome_svg(
        560,
        height,
        &format!(
            r##"<text x="40" y="42" font-family="Segoe UI, Arial, sans-serif" font-size="20" fill="{FG}">svgview keys</text>
  {rows}"##
        ),
    )
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.lines() {
        let mut line = String::new();
        for word in para.split_whitespace() {
            if !line.is_empty() && line.len() + 1 + word.len() > width {
                out.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Document::synthetic` panics on malformed input, so every generator
    /// that feeds it has to be covered.
    #[test]
    fn synthetic_documents_parse() {
        for svg in [welcome_svg(), help_svg(), message_svg("Oops", "something")] {
            let doc = Document::synthetic(&svg);
            let (w, h) = doc.size();
            assert!(w > 0.0 && h > 0.0, "zero-sized synthetic document");
        }
    }

    /// Error text is arbitrary OS-supplied data — a path containing `<` or `&`
    /// must not be able to produce unparseable XML.
    #[test]
    fn message_escapes_markup() {
        let nasty = r#"C:\a&b\<script>x</script>.svg: not found"#;
        let svg = message_svg("Could not open", nasty);
        assert!(!svg.contains("<script>"));
        let doc = Document::synthetic(&svg);
        assert!(doc.size().1 > 0.0);
    }

    #[test]
    fn long_messages_wrap_instead_of_overflowing() {
        let long = "word ".repeat(200);
        let svg = message_svg("Could not open", &long);
        let tall = Document::synthetic(&svg).size().1;
        let short = Document::synthetic(&message_svg("Could not open", "word"))
            .size()
            .1;
        assert!(tall > short, "wrapped message did not grow the document");
    }

    #[test]
    fn text_detection_gates_font_loading() {
        assert!(may_contain_text(br#"<svg><text x="0">hi</text></svg>"#));
        assert!(may_contain_text(br#"<svg><tspan>hi</tspan></svg>"#));
        assert!(may_contain_text(br#"<svg style="font-family:serif"/>"#));
        assert!(!may_contain_text(
            br#"<svg><rect width="1" height="1"/></svg>"#
        ));
        // Gzipped input is opaque to the scan, so it must fail safe to `true`.
        assert!(may_contain_text(&[0x1f, 0x8b, 0x08, 0x00]));
    }

    #[test]
    fn wrap_respects_width_and_keeps_all_words() {
        let lines = wrap("aaa bbb ccc ddd", 7);
        assert!(lines.iter().all(|l| l.len() <= 7), "{lines:?}");
        assert_eq!(lines.join(" "), "aaa bbb ccc ddd");
    }

    #[test]
    fn wrap_never_returns_empty() {
        assert_eq!(wrap("", 10).len(), 1);
    }

    #[test]
    fn loads_a_real_file_and_reports_no_change() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/test.svg"));
        let doc = Document::load(path).expect("assets/test.svg should load");
        assert_eq!(doc.size(), (320.0, 200.0));
        assert_eq!(doc.title(), "test.svg");
        assert!(!doc.changed_on_disk());
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        assert!(Document::load(Path::new("does-not-exist.svg")).is_err());
    }
}

#[cfg(test)]
mod dump {
    /// Development helper, not an assertion: writes the generated chrome to
    /// `target/chrome/` so it can be eyeballed after editing the artwork.
    ///
    ///     cargo test -- --ignored dump_chrome
    #[test]
    #[ignore = "writes files; run explicitly when changing the chrome artwork"]
    fn dump_chrome() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/chrome");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("welcome.svg"), super::welcome_svg()).unwrap();
        std::fs::write(dir.join("help.svg"), super::help_svg()).unwrap();
        std::fs::write(
            dir.join("error.svg"),
            super::message_svg("Could not open", "C:\\Users\\oskar\\Desktop\\chart.svg: The system cannot find the file specified. (os error 2)"),
        ).unwrap();
    }
}
