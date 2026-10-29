import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { C, F } from "./theme";

// Rises and fades in at `at`; fades out over the last `out` frames of `until`.
export const Rise: React.FC<{ at?: number; until?: number; out?: number; dy?: number; children: React.ReactNode; style?: React.CSSProperties }> = ({
  at = 0, until, out = 10, dy = 40, children, style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - at, fps, config: { damping: 200 }, durationInFrames: 18 });
  const fadeOut = until === undefined ? 1 : interpolate(frame, [until - out, until], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return <div style={{ opacity: s * fadeOut, transform: `translateY(${(1 - s) * dy}px)`, ...style }}>{children}</div>;
};

export const Wordmark: React.FC<{ size: number; color?: string }> = ({ size, color = C.text }) => (
  <span style={{ fontFamily: F.serif, fontSize: size, color, letterSpacing: -size * 0.01, lineHeight: 1 }}>
    aeyu<span style={{ color: C.accent }}>.io</span>
  </span>
);

// Caption above the phone: one serif headline, one supporting line.
export const Caption: React.FC<{ title: string; sub?: string; dur: number }> = ({ title, sub, dur }) => (
  <div style={{ position: "absolute", top: 70, left: 60, right: 60, textAlign: "center" }}>
    <Rise at={4} until={dur} out={8}>
      <div style={{ fontFamily: F.serif, fontSize: 82, color: C.text, lineHeight: 1.0 }}>{title}</div>
    </Rise>
    {sub && (
      <Rise at={10} until={dur} out={8}>
        <div style={{ fontFamily: F.sans, fontSize: 36, color: C.text2, marginTop: 14 }}>{sub}</div>
      </Rise>
    )}
  </div>
);
