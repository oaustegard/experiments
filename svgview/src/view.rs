//! Pan/zoom state and rasterisation into a `tiny_skia::Pixmap`.
//!
//! Every redraw re-runs the vector rasteriser at the current zoom, so the
//! output is resolution-correct rather than a scaled bitmap. That is the whole
//! point of a vector viewer, and it is cheap enough to do on resize.

use resvg::tiny_skia::{Pixmap, PremultipliedColorU8, Transform};
use resvg::usvg;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Background {
    Checker,
    White,
    Black,
    Gray,
}

impl Background {
    pub fn next(self) -> Self {
        match self {
            Background::Checker => Background::White,
            Background::White => Background::Black,
            Background::Black => Background::Gray,
            Background::Gray => Background::Checker,
        }
    }
}

pub struct View {
    pub zoom: f32,
    /// Translation in physical pixels, applied after scaling.
    pub tx: f32,
    pub ty: f32,
    pub bg: Background,
    /// While true, a resize re-fits instead of preserving the current zoom.
    pub auto_fit: bool,
}

impl Default for View {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            tx: 0.0,
            ty: 0.0,
            bg: Background::Checker,
            auto_fit: true,
        }
    }
}

const MIN_ZOOM: f32 = 0.01;
const MAX_ZOOM: f32 = 256.0;

impl View {
    pub fn fit(&mut self, doc: (f32, f32), surface: (u32, u32), margin: f32) {
        let (dw, dh) = doc;
        let (sw, sh) = (surface.0 as f32, surface.1 as f32);
        if dw <= 0.0 || dh <= 0.0 || sw <= 0.0 || sh <= 0.0 {
            return;
        }
        let avail_w = (sw - 2.0 * margin).max(1.0);
        let avail_h = (sh - 2.0 * margin).max(1.0);
        self.zoom = (avail_w / dw).min(avail_h / dh).clamp(MIN_ZOOM, MAX_ZOOM);
        self.center(doc, surface);
        self.auto_fit = true;
    }

    /// 1 SVG user unit == 1 logical pixel, i.e. what the SVG author drew.
    pub fn actual_size(&mut self, doc: (f32, f32), surface: (u32, u32), scale: f32) {
        self.zoom = scale;
        self.center(doc, surface);
        self.auto_fit = false;
    }

    pub fn center(&mut self, doc: (f32, f32), surface: (u32, u32)) {
        self.tx = (surface.0 as f32 - doc.0 * self.zoom) / 2.0;
        self.ty = (surface.1 as f32 - doc.1 * self.zoom) / 2.0;
    }

    /// Zoom by `factor`, keeping the document point under `anchor` (physical
    /// pixels) pinned in place.
    pub fn zoom_at(&mut self, anchor: (f32, f32), factor: f32) {
        let old = self.zoom;
        let new = (old * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        if (new - old).abs() < f32::EPSILON {
            return;
        }
        let k = new / old;
        self.tx = anchor.0 - (anchor.0 - self.tx) * k;
        self.ty = anchor.1 - (anchor.1 - self.ty) * k;
        self.zoom = new;
        self.auto_fit = false;
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.tx += dx;
        self.ty += dy;
        self.auto_fit = false;
    }

    pub fn transform(&self) -> Transform {
        Transform::from_translate(self.tx, self.ty).pre_scale(self.zoom, self.zoom)
    }
}

/// Rasterise `tree` into a fresh pixmap of `w`x`h` physical pixels.
///
/// `checker_px` is the checkerboard cell size in physical pixels; pass the
/// DPI-scaled value so the pattern looks the same on a 4K laptop panel.
pub fn rasterize(
    tree: &usvg::Tree,
    view: &View,
    w: u32,
    h: u32,
    checker_px: u32,
) -> Option<Pixmap> {
    let mut pixmap = Pixmap::new(w.max(1), h.max(1))?;
    paint_background(&mut pixmap, view.bg, checker_px.max(2));
    resvg::render(tree, view.transform(), &mut pixmap.as_mut());
    Some(pixmap)
}

/// Rasterise onto full transparency — for PNG export, where baking in a
/// checkerboard would be actively wrong.
pub fn rasterize_transparent(tree: &usvg::Tree, view: &View, w: u32, h: u32) -> Option<Pixmap> {
    let mut pixmap = Pixmap::new(w.max(1), h.max(1))?;
    resvg::render(tree, view.transform(), &mut pixmap.as_mut());
    Some(pixmap)
}

fn paint_background(pixmap: &mut Pixmap, bg: Background, cell: u32) {
    let solid = match bg {
        Background::White => Some((0xff, 0xff, 0xff)),
        Background::Black => Some((0x00, 0x00, 0x00)),
        Background::Gray => Some((0x2b, 0x2b, 0x2b)),
        Background::Checker => None,
    };

    let width = pixmap.width();
    let pixels = pixmap.pixels_mut();

    if let Some((r, g, b)) = solid {
        // Opaque, so premultiplied == straight.
        let px = PremultipliedColorU8::from_rgba(r, g, b, 0xff).unwrap();
        pixels.fill(px);
        return;
    }

    let light = PremultipliedColorU8::from_rgba(0x50, 0x50, 0x50, 0xff).unwrap();
    let dark = PremultipliedColorU8::from_rgba(0x3c, 0x3c, 0x3c, 0xff).unwrap();
    for (i, px) in pixels.iter_mut().enumerate() {
        let x = (i as u32 % width) / cell;
        let y = (i as u32 / width) / cell;
        *px = if (x + y) % 2 == 0 { light } else { dark };
    }
}

/// Convert a premultiplied-RGBA pixmap into softbuffer's 0RGB `u32` words.
///
/// The pixmap has already been composited over an opaque background, so we can
/// demultiply by simply ignoring alpha.
pub fn blit(pixmap: &Pixmap, out: &mut [u32]) {
    for (dst, src) in out.iter_mut().zip(pixmap.pixels()) {
        *dst = (src.red() as u32) << 16 | (src.green() as u32) << 8 | src.blue() as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> (f32, f32) {
        (200.0, 100.0)
    }

    #[test]
    fn fit_leaves_the_margin_and_centres() {
        let mut v = View::default();
        v.fit(doc(), (400, 400), 0.0);
        // Width-limited: 400/200 == 2.0, which also fits the height.
        assert_eq!(v.zoom, 2.0);
        assert_eq!(v.tx, 0.0);
        assert_eq!(v.ty, (400.0 - 200.0) / 2.0);
        assert!(v.auto_fit);
    }

    #[test]
    fn fit_is_height_limited_when_the_window_is_wide() {
        let mut v = View::default();
        v.fit(doc(), (4000, 400), 0.0);
        assert_eq!(v.zoom, 4.0);
    }

    #[test]
    fn fit_ignores_degenerate_sizes() {
        let mut v = View {
            zoom: 3.0,
            ..View::default()
        };
        v.fit((0.0, 0.0), (400, 400), 0.0);
        assert_eq!(
            v.zoom, 3.0,
            "a zero-sized document must not change the zoom"
        );
    }

    /// The point under the cursor has to stay under the cursor — that is the
    /// whole contract of wheel zoom.
    #[test]
    fn zoom_at_pins_the_anchor() {
        let mut v = View {
            zoom: 1.0,
            tx: 30.0,
            ty: -12.0,
            ..View::default()
        };
        let anchor = (250.0, 175.0);
        let before = ((anchor.0 - v.tx) / v.zoom, (anchor.1 - v.ty) / v.zoom);
        v.zoom_at(anchor, 1.25);
        let after = ((anchor.0 - v.tx) / v.zoom, (anchor.1 - v.ty) / v.zoom);
        assert!((before.0 - after.0).abs() < 1e-3, "{before:?} vs {after:?}");
        assert!((before.1 - after.1).abs() < 1e-3, "{before:?} vs {after:?}");
        assert!(!v.auto_fit, "an explicit zoom must cancel auto-fit");
    }

    #[test]
    fn zoom_is_clamped_at_both_ends() {
        let mut v = View::default();
        for _ in 0..200 {
            v.zoom_at((0.0, 0.0), 2.0);
        }
        assert_eq!(v.zoom, MAX_ZOOM);
        for _ in 0..400 {
            v.zoom_at((0.0, 0.0), 0.5);
        }
        assert_eq!(v.zoom, MIN_ZOOM);
        // Clamped at the limit, the anchor maths must not produce NaN.
        assert!(v.tx.is_finite() && v.ty.is_finite());
    }

    #[test]
    fn backgrounds_cycle_back_to_the_start() {
        let mut bg = Background::Checker;
        for _ in 0..4 {
            bg = bg.next();
        }
        assert!(bg == Background::Checker);
    }

    #[test]
    fn rasterize_fills_every_pixel_opaquely() {
        let tree = crate::doc::Document::synthetic(&crate::doc::welcome_svg()).tree;
        let v = View::default();
        let pixmap = rasterize(&tree, &v, 64, 48, 8).expect("allocation");
        assert_eq!(pixmap.width() * pixmap.height(), 64 * 48);
        assert!(
            pixmap.pixels().iter().all(|p| p.alpha() == 255),
            "the background must leave the pixmap fully opaque"
        );
    }

    #[test]
    fn transparent_rasterize_keeps_the_alpha_channel() {
        let tree = crate::doc::Document::synthetic(&crate::doc::welcome_svg()).tree;
        let v = View {
            zoom: 0.1,
            ..View::default()
        };
        let pixmap = rasterize_transparent(&tree, &v, 64, 48).expect("allocation");
        assert!(
            pixmap.pixels().iter().any(|p| p.alpha() == 0),
            "export must not bake in a background"
        );
    }

    #[test]
    fn blit_drops_alpha_and_packs_0rgb() {
        let mut pixmap = Pixmap::new(2, 1).unwrap();
        pixmap
            .pixels_mut()
            .fill(PremultipliedColorU8::from_rgba(0x12, 0x34, 0x56, 0xff).unwrap());
        let mut out = [0u32; 2];
        blit(&pixmap, &mut out);
        assert_eq!(out, [0x00123456, 0x00123456]);
    }
}
