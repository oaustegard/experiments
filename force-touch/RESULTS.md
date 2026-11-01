# Force touch: results

## Answer

Can a browser on an iPhone tell how hard a thumb presses? Not directly. In Chrome on iOS 27.2 (WebKit), both pressure readings stayed at 0 on every press: `PointerEvent.pressure` <!-- @evidence[[results/summary-THUMB7.json#channels.pressure.values.0]] --> and `Touch.force` <!-- @evidence[[results/summary-THUMB7.json#channels.force.values.0]] -->. The phone does report contact size, in coarse steps of 12.139 px of radius. In one posture, size follows force: every firm press read a radius of 36.417 <!-- @evidence[[results/summary-THUMB7.json#channels.radiusX.firm.mean]] --> and every light one 24.278 <!-- @evidence[[results/summary-THUMB7.json#channels.radiusX.light.mean]] -->, an AUC of 1.0 <!-- @evidence[[results/summary-THUMB7.json#channels.radiusX.auc_firm_vs_light]] -->. Change how the thumb lies and the order flips. A firm press with the tip read 12.139 <!-- @evidence[[results/summary-THUMB7.json#channels.radiusX.tip-firm.mean]] -->, smaller than a light flat thumb at 30.348 <!-- @evidence[[results/summary-THUMB7.json#channels.radiusX.flat-light.mean]] -->, so the posture AUC is 0.0 <!-- @evidence[[results/summary-THUMB7.json#channels.radiusX.auc_tip_firm_vs_flat_light]] -->. Contact size measures how flat the thumb is, which tracks force only while posture stays put. For Tilt Rally that makes it at best a two-step hint, light or firm, under a position-based brake. More phones (Safari, Android), more presses per condition, or a thumb that keeps one posture during a race could change this.

## Findings

1. **No pressure channel.** `pressure` and `force` read 0 on all 13 presses (`results/summary-THUMB7.json`, channels.pressure, channels.force).
2. **Contact size is reported, quantised to multiples of 12.139 px.** Every `radiusX`/`radiusY` value seen was 12.139, 24.278 or 36.417, and `width`/`height` came in steps of twice that, up to 145.667 (raw-THUMB7.jsonl).
3. **In one posture, size separates light from firm.** The radius was 24.278 on all three light presses and 36.417 on all three firm ones (AUC 1.0, 3 against 3); medium presses fell in between and overlapped both (summary, channels.radiusX).
4. **Posture outweighs force.** Tip-firm read 12.139 and flat-light 30.348, a posture AUC of 0.0 on radius and 0.25 on width. Size alone would read a firm fingertip as the lightest press of the run (summary, auc_tip_firm_vs_flat_light).
5. **Motion hints, too weak to use.** The phone's z-acceleration while held moved +0.036 m/s² on firm presses against −0.058 on light ones (AUC 1.0 on 3 against 3), but the effect is a few hundredths of a m/s² against a jitter of about 0.1, and it gave 0.75 on the posture pair. The rotation jolt at press onset grew from light to firm (AUC 0.833) (summary, channels.motion_dz, channels.onset_rotation).

## Method

One iPhone, Chrome 156 on iOS 27.2, held sideways in both hands, right thumb on a 120×190 px pad (`web/index.html`). There were 13 prompted presses, each to be held 2 s: light, medium and firm interleaved three times each, then flat-thumb-light and tip-of-thumb-firm twice each, to separate area caused by force from area caused by posture. Each press streamed through the rally-lobby room to `observe.mjs` in a cloud session. `analyze.py` reduces each press to the median of each channel over the hold, excluding lift-off events and the first and last 150 ms. AUC is the Mann–Whitney probability that the firmer condition reads higher. The motion features are the change in mean z-acceleration (gravity removed, and separately with gravity) from the 0.5 s before the press to the hold, and the peak acceleration and rotation rate in the first 300 ms.

## Log

### Run 1 (2026-11-01, room THUMB7)

Oskar ran the 13 prompts on his iPhone while this session observed. All 13 presses arrived once each; only the phone's hello came twice. Oskar said he may have fumbled the first touches. In the data, press 1 (light) lasted 0.98 s instead of 2 s, and press 3 (firm) was held for 8.55 s and arrived thinned to half its samples. Neither changes a finding: press 1 read the same radius as the other two light presses (24.278), and press 3 the same as the other two firm ones (36.417). Press 3 has no motion features. The page keeps at most 10 s of motion and trims to the last 5 s, so the samples from before that long press had been dropped by the time it ended, and there was no baseline to compare against.

The raw radius values were only 12.139, 24.278 and 36.417. That is the WebKit contact estimate quantised to one step per finger-width band, which caps the resolution at about three levels across a natural range of presses. The two flat-light presses read 36.417 and 24.278, so even the same instruction landed on different steps.
