import React from "react";
import { Composition } from "remotion";
import { Reel, REEL_DUR } from "./Reel";
import { FPS } from "./theme";

export const Root: React.FC = () => (
  <Composition id="Reel" component={Reel} durationInFrames={REEL_DUR} fps={FPS} width={1080} height={1920} defaultProps={{ vo: "vo/vo_gemini.wav" }} />
);
