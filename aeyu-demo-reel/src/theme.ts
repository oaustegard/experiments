import { loadFont } from "@remotion/fonts";
import { staticFile } from "remotion";

// Fonts are vendored in public/fonts: the render browser does not trust the
// egress proxy's CA, so fonts.gstatic.com fails inside a render, and a local
// file also keeps the render independent of the network.
const font = (family: string, file: string, weight = "400", style = "normal") =>
  loadFont({ family, url: staticFile(`fonts/${file}`), weight, style });
font("Instrument Serif", "InstrumentSerif-400.woff2");
font("Instrument Serif", "InstrumentSerif-400-italic.woff2", "400", "italic");
font("DM Sans", "DMSans-var.woff2", "400 700");
font("IBM Plex Mono", "IBMPlexMono-400.woff2");

// aeyu.io's own tokens (app.css :root), so the reel and the app read as one brand.
export const C = {
  bg: "#F6F3EE",
  text: "#1A1610",
  text2: "#5C5548",
  text3: "#8C8374",
  onDark: "#FAF7F2",
  accent: "#B85A28",
  accentDark: "#6B3518",
  bezel: "#15120D",
};

export const F = {
  serif: "'Instrument Serif', serif",
  serifItalic: "'Instrument Serif', serif",
  sans: "'DM Sans', sans-serif",
  mono: "'IBM Plex Mono', monospace",
};

export const FPS = 30;
// iPhone 14 viewport, in CSS px; captures are @3x.
export const VIEW = { w: 390, h: 844 };
