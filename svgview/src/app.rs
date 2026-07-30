//! The winit application: window, input, and the redraw loop.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

use crate::doc::{self, Document};
use crate::view::{rasterize, rasterize_transparent, View};

/// How often to stat the open file for the auto-reload.
const POLL: Duration = Duration::from_millis(300);
/// Gap left around the drawing when fitting to the window, in logical pixels.
const FIT_MARGIN: f32 = 12.0;
const ZOOM_STEP: f32 = 1.25;
const PAN_STEP: f32 = 48.0;

pub struct App {
    window: Option<Arc<Window>>,
    context: Option<softbuffer::Context<Arc<Window>>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,

    doc: Document,
    /// The real document, parked while the help overlay is showing.
    stashed: Option<Document>,
    view: View,

    modifiers: ModifiersState,
    cursor: (f32, f32),
    dragging: bool,
    scale: f32,
    next_poll: Instant,
    /// Set when the window should re-fit on the next redraw (initial show,
    /// new file, reload with different dimensions).
    refit: bool,
}

impl App {
    pub fn new(initial: Option<PathBuf>) -> Self {
        let (doc, error) = match initial {
            Some(path) => match Document::load(&path) {
                Ok(d) => (d, None),
                Err(e) => (
                    Document::synthetic(&doc::message_svg("Could not open", &e)),
                    Some(e),
                ),
            },
            None => (Document::synthetic(&doc::welcome_svg()), None),
        };
        if let Some(e) = &error {
            eprintln!("svgview: {e}");
        }
        Self {
            window: None,
            context: None,
            surface: None,
            doc,
            stashed: None,
            view: View::default(),
            modifiers: ModifiersState::empty(),
            cursor: (0.0, 0.0),
            dragging: false,
            scale: 1.0,
            next_poll: Instant::now() + POLL,
            refit: true,
        }
    }

    fn surface_size(&self) -> (u32, u32) {
        self.window
            .as_ref()
            .map(|w| {
                let s = w.inner_size();
                (s.width.max(1), s.height.max(1))
            })
            .unwrap_or((1, 1))
    }

    fn set_title(&self) {
        if let Some(window) = &self.window {
            let pct = (self.view.zoom / self.scale * 100.0).round() as i32;
            window.set_title(&format!("{} — {}% — svgview", self.doc.title(), pct));
        }
    }

    fn request_redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn open(&mut self, path: &Path) {
        match Document::load(path) {
            Ok(d) => {
                self.doc = d;
                self.stashed = None;
            }
            Err(e) => {
                eprintln!("svgview: {e}");
                self.doc = Document::synthetic(&doc::message_svg("Could not open", &e));
                self.stashed = None;
            }
        }
        self.refit = true;
        self.request_redraw();
    }

    fn reload(&mut self) {
        // While help is up the real document is stashed; reload that one.
        let target = self.stashed.as_mut().unwrap_or(&mut self.doc);
        if target.path.is_none() {
            return;
        }
        let before = target.size();
        match target.reload() {
            Ok(()) => {
                if target.size() != before {
                    self.refit = true;
                }
            }
            Err(e) => {
                eprintln!("svgview: {e}");
                let msg = Document::synthetic(&doc::message_svg("Reload failed", &e));
                if self.stashed.is_some() {
                    self.stashed = Some(msg);
                } else {
                    self.doc = msg;
                    self.refit = true;
                }
            }
        }
        self.request_redraw();
    }

    fn toggle_help(&mut self) {
        match self.stashed.take() {
            Some(previous) => self.doc = previous,
            None => {
                self.stashed = Some(std::mem::replace(
                    &mut self.doc,
                    Document::synthetic(&doc::help_svg()),
                ));
            }
        }
        self.refit = true;
        self.request_redraw();
    }

    fn pick_file(&mut self) {
        #[cfg(windows)]
        {
            let start = self
                .doc
                .path
                .as_ref()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf());
            let mut dialog = rfd::FileDialog::new().add_filter("SVG", &["svg", "svgz"]);
            if let Some(dir) = start {
                dialog = dialog.set_directory(dir);
            }
            if let Some(path) = dialog.pick_file() {
                self.open(&path);
            }
        }
        #[cfg(not(windows))]
        {
            // No dialog outside Windows by design — see Cargo.toml. Dropping a
            // file on the window and the command-line argument both still work.
            eprintln!("svgview: pass a file path on the command line, or drop one on the window");
        }
    }

    fn export_png(&mut self) {
        let (w, h) = self.surface_size();
        let Some(pixmap) = rasterize_transparent(&self.doc.tree, &self.view, w, h) else {
            return;
        };
        let out = match &self.doc.path {
            Some(p) => p.with_extension("png"),
            None => PathBuf::from("svgview-export.png"),
        };
        match pixmap.save_png(&out) {
            Ok(()) => eprintln!("svgview: wrote {}", out.display()),
            Err(e) => eprintln!("svgview: could not write {}: {e}", out.display()),
        }
    }

    fn toggle_fullscreen(&self) {
        if let Some(window) = &self.window {
            let next = match window.fullscreen() {
                Some(_) => None,
                None => Some(Fullscreen::Borderless(None)),
            };
            window.set_fullscreen(next);
        }
    }

    fn on_key(&mut self, key: Key, event_loop: &ActiveEventLoop) {
        let ctrl = self.modifiers.control_key();
        let size = self.surface_size();
        let doc_size = self.doc.size();

        match key.as_ref() {
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Character("w") if ctrl => event_loop.exit(),
            Key::Character("o") if ctrl => self.pick_file(),
            Key::Character("s") if ctrl => self.export_png(),
            Key::Character("0") if ctrl => {
                self.view.fit(doc_size, size, FIT_MARGIN * self.scale);
            }
            Key::Character("1") if ctrl => {
                self.view.actual_size(doc_size, size, self.scale);
            }
            Key::Character("+" | "=") if ctrl => {
                let c = (size.0 as f32 / 2.0, size.1 as f32 / 2.0);
                self.view.zoom_at(c, ZOOM_STEP);
            }
            Key::Character("-" | "_") if ctrl => {
                let c = (size.0 as f32 / 2.0, size.1 as f32 / 2.0);
                self.view.zoom_at(c, 1.0 / ZOOM_STEP);
            }
            Key::Character("b") => {
                self.view.bg = self.view.bg.next();
            }
            Key::Character("r") => self.reload(),
            Key::Character("h") => self.toggle_help(),
            Key::Named(NamedKey::F11) => self.toggle_fullscreen(),
            Key::Named(NamedKey::ArrowLeft) => self.view.pan(PAN_STEP * self.scale, 0.0),
            Key::Named(NamedKey::ArrowRight) => self.view.pan(-PAN_STEP * self.scale, 0.0),
            Key::Named(NamedKey::ArrowUp) => self.view.pan(0.0, PAN_STEP * self.scale),
            Key::Named(NamedKey::ArrowDown) => self.view.pan(0.0, -PAN_STEP * self.scale),
            _ => return,
        }
        self.set_title();
        self.request_redraw();
    }

    fn redraw(&mut self) {
        let (w, h) = self.surface_size();
        if self.refit {
            self.view
                .fit(self.doc.size(), (w, h), FIT_MARGIN * self.scale);
            self.refit = false;
            self.set_title();
        } else if self.view.auto_fit {
            // Keep a fitted document fitted across resizes.
            self.view
                .fit(self.doc.size(), (w, h), FIT_MARGIN * self.scale);
        }

        let (Some(surface), Some(window)) = (self.surface.as_mut(), self.window.as_ref()) else {
            return;
        };
        let (Some(nw), Some(nh)) = (NonZeroU32::new(w), NonZeroU32::new(h)) else {
            return;
        };
        if surface.resize(nw, nh).is_err() {
            return;
        }
        let Ok(mut buffer) = surface.buffer_mut() else {
            return;
        };

        let checker = (8.0 * self.scale).round().max(2.0) as u32;
        match rasterize(&self.doc.tree, &self.view, w, h, checker) {
            Some(pixmap) => crate::view::blit(&pixmap, &mut buffer),
            // Allocation failed (absurd window size); leave the buffer alone
            // rather than presenting garbage.
            None => return,
        }
        window.pre_present_notify();
        let _ = buffer.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("svgview")
            .with_inner_size(LogicalSize::new(1000.0, 720.0));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("svgview: could not create a window: {e}");
                event_loop.exit();
                return;
            }
        };
        self.scale = window.scale_factor() as f32;

        match softbuffer::Context::new(window.clone())
            .and_then(|ctx| softbuffer::Surface::new(&ctx, window.clone()).map(|s| (ctx, s)))
        {
            Ok((ctx, surface)) => {
                self.context = Some(ctx);
                self.surface = Some(surface);
            }
            Err(e) => {
                eprintln!("svgview: could not create a drawing surface: {e}");
                event_loop.exit();
                return;
            }
        }

        self.window = Some(window);
        self.refit = true;
        self.set_title();
        self.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::Resized(_) => self.request_redraw(),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale = scale_factor as f32;
                self.refit = true;
                self.request_redraw();
            }
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::DroppedFile(path) => self.open(&path),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                self.on_key(event.logical_key, event_loop);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let next = (position.x as f32, position.y as f32);
                if self.dragging {
                    self.view
                        .pan(next.0 - self.cursor.0, next.1 - self.cursor.1);
                    self.request_redraw();
                }
                self.cursor = next;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state == ElementState::Pressed;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    // Trackpads report pixels; 120px ≈ one detent.
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                };
                if steps != 0.0 {
                    self.view.zoom_at(self.cursor, ZOOM_STEP.powf(steps));
                    self.set_title();
                    self.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        if now >= self.next_poll {
            self.next_poll = now + POLL;
            let watched = self.stashed.as_ref().unwrap_or(&self.doc);
            if watched.changed_on_disk() {
                self.reload();
            }
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_poll));
    }
}
