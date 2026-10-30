import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig, staticFile, Img, Html5Audio } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { C, F } from "./theme";
import { Phone, PHONE_W, PHONE_H, K } from "./Phone";
import { ScrollShot, scrollAt, Key } from "./ScrollShot";
import { FrameSequence } from "./FrameSequence";
import { Caption, Rise, Wordmark } from "./Text";
import { Tap } from "./Tap";
import layout from "../public/shots/layout.json";

const HEADER = layout.header.h; // sticky app header, CSS px
const T = 15; // transition length, frames

// ---------- phone shots ----------
const dashKeys: Key[] = [{ f: 10, y: 0 }, { f: 95, y: 1000 }];
const tapY = layout.dashRiverLoop.y + 60 - 1000; // viewport y of River Loop's title
const actKeys: Key[] = [{ f: 8, y: 0 }, { f: 60, y: layout.actAwards.y - HEADER - 20 }];

type Shot = { id: string; dur: number; title: string; sub?: string; body: React.FC };
const SHOTS: Shot[] = [
  {
    id: "dashboard", dur: 150, title: "Awards for the rest of us",
    sub: "Year bests, season firsts, streaks",
    body: () => (
      <ScrollShot src="shots/dashboard.png" keys={dashKeys} stickyH={HEADER}>
        <Tap x={(layout.dashRiverLoop.x + layout.dashRiverLoop.w / 2) * K} y={tapY * K} at={118} />
      </ScrollShot>
    ),
  },
  {
    id: "replay", dur: 150, title: "Ride it again", sub: "A 3D replay of every ride, colored by power",
    body: () => <FrameSequence dir="shots/replay" count={150} />,
  },
  {
    id: "awards", dur: 150, title: "50 ways to be proud", sub: "Computed per ride, from your own history",
    body: () => <ScrollShot src="shots/activity.png" keys={actKeys} stickyH={HEADER} />,
  },
  {
    id: "power", dur: 135, title: "Power that knows you're tired", sub: "Your curve after 1,000, 2,000, 3,000 kJ",
    body: () => <ScrollShot src="shots/power.png" keys={[{ f: 20, y: 0 }, { f: 120, y: 260 }]} stickyH={HEADER} />,
  },
  {
    id: "heatmap", dur: 135, title: "Every ride, one map", sub: "Watch your years fill in",
    body: () => <FrameSequence dir="shots/heatmap-play" count={135} />,
  },
];
const PHONE_DUR = SHOTS.reduce((a, s) => a + s.dur, 0) - T * (SHOTS.length - 1);

const PhoneSection: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const enter = spring({ frame, fps, config: { damping: 18, mass: 0.9 }, durationInFrames: 30 });
  const exit = interpolate(frame, [PHONE_DUR - 18, PHONE_DUR], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const SCALE = 0.86;
  const y = 330;
  return (
    <AbsoluteFill>
      <TransitionSeries>
        {SHOTS.flatMap((s, i) => [
          <TransitionSeries.Sequence key={s.id + "c"} durationInFrames={s.dur}>
            <Caption title={s.title} sub={s.sub} dur={s.dur} />
          </TransitionSeries.Sequence>,
          i < SHOTS.length - 1 ? <TransitionSeries.Transition key={s.id + "ct"} presentation={fade()} timing={linearTiming({ durationInFrames: T })} /> : null,
        ])}
      </TransitionSeries>
      <div style={{ position: "absolute", left: (1080 - PHONE_W) / 2, top: y + (1 - enter) * 900 + exit * 40, opacity: 1 - exit, transform: `scale(${SCALE * (1 - exit * 0.04)})`, transformOrigin: "top center" }}>
        <Phone statusBg={C.accentDark}>
          <TransitionSeries>
            {SHOTS.flatMap((s, i) => [
              <TransitionSeries.Sequence key={s.id} durationInFrames={s.dur}>
                <AbsoluteFill style={{ background: C.bg }}><s.body /></AbsoluteFill>
              </TransitionSeries.Sequence>,
              i < SHOTS.length - 1 ? (
                <TransitionSeries.Transition
                  key={s.id + "t"}
                  // replay -> awards is the same page scrolling, so cross-fade; the rest slide like navigation
                  presentation={s.id === "replay" ? fade() : slide({ direction: "from-right" })}
                  timing={linearTiming({ durationInFrames: T })}
                />
              ) : null,
            ])}
          </TransitionSeries>
        </Phone>
      </div>
      <div style={{ position: "absolute", bottom: 46, width: "100%", textAlign: "center", opacity: enter * (1 - exit) }}>
        <Wordmark size={44} />
      </div>
    </AbsoluteFill>
  );
};

// ---------- text scenes ----------
// Frames at which the opening line spells each character of "aeyu.io",
// measured on public/vo/vo_v7.wav (energy segmentation + whisper word timestamps).
const TITLE_AT = [9, 27, 45, 67, 88, 100, 107]; // a e y u . i o

const SpelledWordmark: React.FC<{ size: number }> = ({ size }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <span style={{ fontFamily: F.serif, fontSize: size, letterSpacing: -size * 0.01, lineHeight: 1 }}>
      {"aeyu.io".split("").map((ch, k) => {
        const hit = spring({ frame: frame - TITLE_AT[k], fps, config: { damping: 14 }, durationInFrames: 16 });
        const glow = interpolate(frame - TITLE_AT[k], [0, 4, 20], [0, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
        const base = k >= 4 ? C.accent : C.text;
        return (
          <span key={k} style={{
            display: "inline-block", color: glow > 0.3 ? C.accent : base,
            opacity: 0.25 + 0.75 * hit, transform: `translateY(${-14 * glow}px)`,
          }}>{ch}</span>
        );
      })}
    </span>
  );
};

const Title: React.FC = () => (
  <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", textAlign: "center" }}>
    <Rise at={0} dy={20}><SpelledWordmark size={210} /></Rise>
    <Rise at={118}>
      <div style={{ fontFamily: F.serifItalic, fontStyle: "italic", fontSize: 50, color: C.text3, marginTop: 30, padding: "0 80px" }}>
        The sound you make at the top of the climb
      </div>
    </Rise>
  </AbsoluteFill>
);

const Hook: React.FC = () => (
  <AbsoluteFill style={{ justifyContent: "center", padding: "0 90px" }}>
    <Rise at={2}>
      <div style={{ fontFamily: F.serif, fontSize: 104, color: C.text3, lineHeight: 1.02 }}>Strava crowns the fastest.</div>
    </Rise>
    <Rise at={42}>
      <div style={{ fontFamily: F.serif, fontSize: 104, color: C.text, lineHeight: 1.02, marginTop: 50 }}>
        aeyu<span style={{ color: C.accent }}>.io</span> celebrates the rest of your rides.
      </div>
    </Rise>
  </AbsoluteFill>
);

// Reads follow the narration: "No ads, no tracking." / "Just you and your Strava
// data," / "on your device, in your browser."
const Privacy: React.FC = () => (
  <AbsoluteFill style={{ justifyContent: "center", padding: "0 90px" }}>
    <Rise at={2}>
      <div style={{ fontFamily: F.mono, fontSize: 34, color: C.accent, letterSpacing: 2, textTransform: "uppercase" }}>No ads · No tracking</div>
    </Rise>
    <Rise at={40}>
      <div style={{ fontFamily: F.serif, fontSize: 110, color: C.text, lineHeight: 1.02, marginTop: 30 }}>
        Just you and <span style={{ color: C.accent }}>your</span> Strava data.
      </div>
    </Rise>
    <Rise at={105}>
      <div style={{ fontFamily: F.sans, fontSize: 44, color: C.text2, marginTop: 44, lineHeight: 1.35 }}>
        On <b style={{ color: C.text }}>your</b> device, in <b style={{ color: C.text }}>your</b> browser. Nothing goes to a server.
      </div>
    </Rise>
  </AbsoluteFill>
);

// Frames (from the closing card's start) at which the narrator says each vowel,
// measured on public/vo/vo_v7.wav (energy segmentation + whisper word timestamps).
const VOWEL_AT: Record<string, number> = { a: 44, e: 51, y: 60, u: 72, i: 96, o: 105 };
const ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];
const ROW_SHIFT = [0, 0.25, 0.75]; // QWERTY stagger, in key widths

// "A, E, Y, U, dot I, O: every vowel on your keyboard, left to right."
const Keyboard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const K = 84, G = 8;
  return (
    <div style={{ position: "relative", width: 10 * K + 9 * G, height: 3 * K + 2 * G }}>
      {ROWS.flatMap((row, r) =>
        row.split("").map((ch, c) => {
          const at = VOWEL_AT[ch];
          const lit = at === undefined ? 0 : spring({ frame: frame - at, fps, config: { damping: 12 }, durationInFrames: 14 });
          return (
            <div key={ch} style={{
              position: "absolute", left: (c + ROW_SHIFT[r]) * (K + G), top: r * (K + G), width: K, height: K, borderRadius: 14,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: F.mono, fontSize: 40, textTransform: "uppercase",
              background: lit > 0.5 ? C.accent : "#FFFFFF", color: lit > 0.5 ? C.onDark : C.text3,
              border: `2px solid ${lit > 0.5 ? C.accentDark : "#E2DBD0"}`,
              transform: `translateY(${-8 * lit}px) scale(${1 + 0.12 * lit})`,
              boxShadow: lit > 0.5 ? "0 10px 24px rgba(184,90,40,0.35)" : "0 2px 0 #E2DBD0",
            }}>{ch}</div>
          );
        })
      )}
    </div>
  );
};

const Cta: React.FC = () => {
  const frame = useCurrentFrame();
  const pulse = 1 + 0.03 * Math.sin(Math.max(0, frame - 30) / 6);
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", textAlign: "center" }}>
      <Rise at={0}><Img src={staticFile("shots/icon.png")} style={{ width: 150, height: 150, borderRadius: 34, boxShadow: "0 20px 50px rgba(107,53,24,0.35)" }} /></Rise>
      <Rise at={4}><div style={{ marginTop: 36 }}><Wordmark size={150} /></div></Rise>
      <Rise at={10}><div style={{ fontFamily: F.serif, fontSize: 54, color: C.text2, marginTop: 6 }}>Participation Awards</div></Rise>
      <Rise at={16}>
        <div style={{ marginTop: 50, transform: `scale(${pulse})`, background: C.accent, color: C.onDark, fontFamily: F.sans, fontWeight: 700, fontSize: 48, padding: "26px 68px", borderRadius: 999 }}>
          All free · aeyu.io
        </div>
      </Rise>
      <Rise at={40}><div style={{ marginTop: 80 }}><Keyboard /></div></Rise>
      <Rise at={150}><div style={{ fontFamily: F.serif, fontSize: 52, color: C.text, marginTop: 44 }}>Every vowel on your keyboard, left to right.</div></Rise>
      <Rise at={185}><div style={{ fontFamily: F.sans, fontSize: 32, color: C.text3, marginTop: 26 }}>No ads. No tracking. Try the demo, or connect Strava.</div></Rise>
    </AbsoluteFill>
  );
};

// ---------- reel ----------
const SCENES: { id: string; dur: number; C: React.FC }[] = [
  { id: "title", dur: 210, C: Title },
  { id: "hook", dur: 120, C: Hook },
  { id: "phone", dur: PHONE_DUR, C: PhoneSection },
  { id: "privacy", dur: 210, C: Privacy },
  { id: "cta", dur: 240, C: Cta },
];
export const REEL_DUR = SCENES.reduce((a, s) => a + s.dur, 0) - T * (SCENES.length - 1);

// `vo` picks the narration track in public/vo/; render another with --props='{"vo":"vo/x.wav"}'
export const Reel: React.FC<{ vo?: string }> = ({ vo = "vo/vo_v7.wav" }) => (
  <AbsoluteFill style={{ background: `radial-gradient(ellipse at 50% 30%, #FBF8F3 0%, ${C.bg} 55%, #EDE6DB 100%)` }}>
    {/* narration, mixed to this timeline by vo/assemble.py */}
    <Html5Audio src={staticFile(vo)} />
    <TransitionSeries>
      {SCENES.flatMap((s, i) => [
        <TransitionSeries.Sequence key={s.id} durationInFrames={s.dur}><s.C /></TransitionSeries.Sequence>,
        i < SCENES.length - 1 ? <TransitionSeries.Transition key={s.id + "t"} presentation={fade()} timing={linearTiming({ durationInFrames: T })} /> : null,
      ])}
    </TransitionSeries>
  </AbsoluteFill>
);
