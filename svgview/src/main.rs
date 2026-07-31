// On Windows, a release build is a GUI app: no console window flashes up when
// it is launched from Explorer or as a file-association handler. Debug builds
// keep the console so `eprintln!` diagnostics are visible while developing.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod doc;
mod view;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use winit::event_loop::{ControlFlow, EventLoop};

const USAGE: &str = "\
svgview — a small native SVG viewer

USAGE:
    svgview [FILE]
    svgview FILE --png OUT [--width PX | --height PX | --scale N]

OPTIONS:
    --png OUT      Render FILE to a PNG and exit; no window is opened.
    --width PX     With --png: output width in pixels (height follows the aspect ratio).
    --height PX    With --png: output height in pixels.
    --scale N      With --png: multiply the SVG's natural size by N. Default 1.
    -h, --help     Show this message.
    -V, --version  Show the version.

VIEWER KEYS:
    Ctrl+O open   wheel zoom   drag/arrows pan   Ctrl+0 fit   Ctrl+1 100%
    B background  R reload     Ctrl+S export PNG  F11 fullscreen  H help
";

enum Mode {
    View(Option<PathBuf>),
    Png {
        input: PathBuf,
        output: PathBuf,
        sizing: Sizing,
    },
}

enum Sizing {
    Scale(f32),
    Width(u32),
    Height(u32),
}

fn main() -> ExitCode {
    let mode = match parse_args(std::env::args().skip(1).collect()) {
        Ok(Some(mode)) => mode,
        // --help / --version already printed.
        Ok(None) => return ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("svgview: {e}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    match mode {
        Mode::View(path) => run_viewer(path),
        Mode::Png {
            input,
            output,
            sizing,
        } => match render_to_png(&input, &output, sizing) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("svgview: {e}");
                ExitCode::FAILURE
            }
        },
    }
}

fn parse_args(args: Vec<String>) -> Result<Option<Mode>, String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut sizing = Sizing::Scale(1.0);
    let mut iter = args.into_iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("svgview {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "--png" => {
                let v = iter.next().ok_or("--png needs an output path")?;
                output = Some(PathBuf::from(v));
            }
            "--width" => sizing = Sizing::Width(parse_dim(iter.next(), "--width")?),
            "--height" => sizing = Sizing::Height(parse_dim(iter.next(), "--height")?),
            "--scale" => {
                let v = iter.next().ok_or("--scale needs a number")?;
                let n: f32 = v
                    .parse()
                    .map_err(|_| format!("--scale: not a number: {v}"))?;
                if !(n.is_finite() && n > 0.0) {
                    return Err(format!("--scale must be positive: {v}"));
                }
                sizing = Sizing::Scale(n);
            }
            other if other.starts_with('-') && other.len() > 1 => {
                return Err(format!("unknown option: {other}"));
            }
            path => {
                if input.is_some() {
                    return Err("only one input file is supported".into());
                }
                input = Some(PathBuf::from(path));
            }
        }
    }

    match (input, output) {
        (Some(input), Some(output)) => Ok(Some(Mode::Png {
            input,
            output,
            sizing,
        })),
        (None, Some(_)) => Err("--png needs an input file".into()),
        (input, None) => Ok(Some(Mode::View(input))),
    }
}

fn parse_dim(value: Option<String>, flag: &str) -> Result<u32, String> {
    let v = value.ok_or_else(|| format!("{flag} needs a pixel count"))?;
    let n: u32 = v
        .parse()
        .map_err(|_| format!("{flag}: not a pixel count: {v}"))?;
    if n == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(n)
}

fn run_viewer(path: Option<PathBuf>) -> ExitCode {
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("svgview: could not start the event loop: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Redraw only in response to input or the reload poll — an idle viewer
    // should use no CPU at all.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = app::App::new(path);
    match event_loop.run_app(&mut app) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("svgview: {e}");
            ExitCode::FAILURE
        }
    }
}

fn render_to_png(input: &Path, output: &Path, sizing: Sizing) -> Result<(), String> {
    let document = doc::Document::load(input)?;
    let (dw, dh) = document.size();
    if dw <= 0.0 || dh <= 0.0 {
        return Err(format!("{}: document has zero size", input.display()));
    }

    let zoom = match sizing {
        Sizing::Scale(n) => n,
        Sizing::Width(px) => px as f32 / dw,
        Sizing::Height(px) => px as f32 / dh,
    };
    let (w, h) = (
        (dw * zoom).round().max(1.0) as u32,
        (dh * zoom).round().max(1.0) as u32,
    );

    let view = view::View {
        zoom,
        tx: 0.0,
        ty: 0.0,
        ..view::View::default()
    };
    let pixmap = view::rasterize_transparent(&document.tree, &view, w, h)
        .ok_or_else(|| format!("could not allocate a {w}x{h} image"))?;
    pixmap
        .save_png(output)
        .map_err(|e| format!("could not write {}: {e}", output.display()))?;
    println!("{} ({}x{})", output.display(), w, h);
    Ok(())
}
