// Fights decided by played-out futures. RolloutBot wraps the trm-doom Bot unchanged: outside a fight it passes the
// Bot's input through. In a fight, every `commit` tics, it forks the game on the GPU and plays each of a set of
// fighting moves for `horizon` tics in `scenarios` copies with the random-number index reseeded, scores the
// outcome from engine state (damage dealt and taken, kills, death), and plays the best move until the next decision.
// Moves are small policies, re-evaluated against the live state each tic: aim at the enemy and fire when on target
// (the Bot's own rule), while moving in a direction relative to the enemy's bearing.

const TAU = 2 * Math.PI;
const wrap = (a) => { a = (a + Math.PI) % TAU; if (a < 0) a += TAU; return a - Math.PI; };

// direction relative to the bearing to the enemy, or null to stand; "route" keeps the Bot's own movement
export const MOVES = [
  ["bot", "route"], ["stand", null], ["toward", 0], ["away", Math.PI], ["left", Math.PI / 2], ["right", -Math.PI / 2],
  ["back-left", 3 * Math.PI / 4], ["back-right", -3 * Math.PI / 4], ["fwd-left", Math.PI / 4], ["fwd-right", -Math.PI / 4],
];

export class RolloutBot {
  constructor(bot, forker, ctx, opts = {}) {
    this.bot = bot; this.fk = forker; this.W = ctx.W; this.MF = ctx.MF; this.eng = ctx.eng;
    this.o = { horizon: 35, commit: 8, scenarios: 2, readEvery: 3, risk: 0.5, wDamage: 1, wKill: 50, wHurt: 4, wDeath: 2000, hysteresis: 5, ...opts };
    this.plan = null;
    this.rstats = { decisions: 0, rolloutTics: 0, ms: 0, chosen: {}, overrides: 0 };
    this.decisions = [];
  }
  get s() { return this.bot.s; }
  get stats() { return { ...this.bot.stats, rollout: { ...this.rstats, meanMs: Math.round(this.rstats.ms / Math.max(1, this.rstats.decisions)) } }; }
  get events() { return this.bot.events; }
  debug() { return { ...this.bot.debug(), move: this.plan?.name }; }
  set debugGrids(v) { this.bot.debugGrids = v; }
  set globalPlanner(v) { this.bot.globalPlanner = v; }

  readState(st) { this.st = st; return this.bot.readState(st); }

  // the move's input for one tic, from the player's position and angle and the enemy's position
  policy(move, base, px, py, pa, ex, ey, er, eAlive) {
    const d = Math.hypot(ex - px, ey - py) || 1;
    const bearing = Math.atan2(ey - py, ex - px);
    const err = wrap(bearing - pa);
    const input = { forward: 0, side: 0, turn: Math.max(-0.3, Math.min(0.3, err)), use: false, weapon: 0,
      fire: eAlive && Math.abs(err) < Math.max(0.05, Math.atan2(er * 0.8, d)) && (this.melee ? d < 80 : d < 900 || this.rocket) };
    let vx = 0, vy = 0;
    if (move[1] === "route") { // the Bot's own movement this decision, held in world terms
      vx = base.vx; vy = base.vy;
    } else if (move[1] !== null) { const a = bearing + move[1]; vx = Math.cos(a); vy = Math.sin(a); }
    const ca = Math.cos(pa), sa = Math.sin(pa);
    input.forward = 50 * (vx * ca + vy * sa);
    input.side = 40 * (vx * sa - vy * ca);
    return input;
  }

  // world-frame movement of a Bot input, given the player's angle (inverse of the mapping above)
  worldMove(inp, pa) {
    const f = (inp.forward || 0) / 50, s = (inp.side || 0) / 40, ca = Math.cos(pa), sa = Math.sin(pa);
    return { vx: f * ca + s * sa, vy: f * sa - s * ca };
  }

  // a move whose first 40 units leave the Bot's own fine grid (a wall, a drop, nukage) is not tried
  safe(move, base, p, en) {
    if (move[1] === "route" || move[1] === null) return true;
    const g = this.bot.fine?.grid;
    if (!g) return true;
    const a = Math.atan2(en.m.y - p.y, en.m.x - p.x) + move[1];
    return this.bot.gridLineClear(g, p.x, p.y, p.x + Math.cos(a) * 40, p.y + Math.sin(a) * 40, true);
  }

  monsterHealth(mobjBuf) {
    const mi = new Int32Array(mobjBuf), mu = new Uint32Array(mobjBuf), MF = this.MF, h = new Map();
    for (let k = 1; k < mu.length / 32; k++) {
      const o = k * 32;
      if (!mu[o + 24] || !(mu[o + 14] & MF.COUNTKILL)) continue;
      h.set(k, Math.max(0, mi[o + 15]));
    }
    return h;
  }

  async rollout(move, base, seed, en, h0, w0) {
    const W = this.W, fk = this.fk, eng = this.eng, o = this.o;
    fk.restore();
    fk.setWord(W.rng, seed);
    let px = this.s.p.x, py = this.s.p.y, pa = this.s.p.angle, ex = en.m.x, ey = en.m.y, alive = true;
    const slot = en.m.k * 128;
    for (let t = 0; t < o.horizon; t++) {
      if (t > 0 && t % o.readEvery === 0) {
        const [wb, mb] = await fk.read([[eng.worldBuf, 0, 16], [eng.mobjBuf, slot, 128]]);
        const wf = new Float32Array(wb), mi = new Int32Array(mb), mu = new Uint32Array(mb);
        px = wf[W.viewx]; py = wf[W.viewy]; pa = wf[W.viewangle];
        ex = mi[0] / 65536; ey = mi[1] / 65536; alive = !!mu[24] && mi[15] > 0;
      }
      const inp = this.policy(move, base, px, py, pa, ex, ey, en.m.r, alive);
      if (t === 0 && base.weapon) inp.weapon = base.weapon;
      fk.tic(inp);
    }
    const [wb, mb] = await fk.read([[eng.worldBuf, 0, 1280], [eng.mobjBuf, 0, eng.mobjBuf.size]]);
    const wi = new Int32Array(wb), wu = new Uint32Array(wb);
    const h1 = this.monsterHealth(mb);
    let dealt = 0;
    for (const [k, v] of h0) dealt += v - (h1.get(k) ?? 0);
    const kills = wi[W.killcount] - w0.kills;
    const hurt = (w0.health - Math.max(0, wi[W.health])) + (w0.armor - wi[W.armor]) / 3;
    const dead = wu[W.dead] ? 1 : 0;
    this.rstats.rolloutTics += o.horizon;
    return o.wDamage * dealt + o.wKill * kills - o.wHurt * hurt - o.wDeath * dead;
  }

  async decide(base, p, en) {
    const t0 = performance.now(), o = this.o;
    this.fk.save();
    const h0 = this.monsterHealth(this.st.mobj);
    const w0 = { kills: p.kills, health: p.health, armor: p.armor };
    const rng0 = new Uint32Array(this.st.world)[this.W.rng];
    const rows = [];
    for (const move of MOVES) {
      if (!this.safe(move, base, p, en)) continue;
      const sc = [];
      // reseeded copies, never the game's own index: the futures could happen, they are not the one that will
      for (let k = 0; k < o.scenarios; k++) sc.push(await this.rollout(move, base, (rng0 + 97 + 61 * k) & 255, en, h0, w0));
      const mean = sc.reduce((a, b) => a + b, 0) / sc.length, worst = Math.min(...sc);
      let score = mean - o.risk * (mean - worst);
      if (this.plan && this.plan.name === move[0]) score += o.hysteresis;
      rows.push({ name: move[0], move, score, sc });
    }
    this.fk.restore();
    rows.sort((a, b) => b.score - a.score);
    const ms = performance.now() - t0;
    this.rstats.decisions++; this.rstats.ms += ms;
    const best = rows[0];
    this.rstats.chosen[best.name] = (this.rstats.chosen[best.name] || 0) + 1;
    if (best.name !== "bot") this.rstats.overrides++;
    if (this.decisions.length < 4000) this.decisions.push({ tic: this.bot.tic, enemy: en.m.k, d: Math.round(en.d), hp: p.health, best: best.name,
      rows: rows.map((r) => [r.name, Math.round(r.score), ...r.sc.map(Math.round)]) });
    return best;
  }

  async step() {
    const input = await this.bot.step();
    const p = this.s.p;
    if (p.dead || p.exit) return input;
    const en = this.bot.enemy();
    // only a fight the Bot would take on: visible, and worth facing with the ammo it has
    const noAmmo = p.ammo[0] <= 0 && p.ammo[1] <= 0 && p.ammo[2] <= 0 && p.ammo[3] <= 0;
    if (!en || en.unseen || (noAmmo && en.d > 120) || input.use || this.bot.unstick > 0) { this.plan = null; return input; }
    // the Bot's fire rule: rockets and plasma from any range, fist and chainsaw only in reach
    this.rocket = p.ready >= 4 && p.ready <= 6; this.melee = p.ready === 0 || p.ready === 7;
    const base = { ...this.worldMove(input, p.angle), weapon: input.weapon || 0 };
    if (!this.plan || this.bot.tic >= this.plan.until || this.plan.enemy !== en.m.k) {
      const best = await this.decide(base, p, en);
      this.plan = { name: best.name, move: best.move, until: this.bot.tic + this.o.commit, enemy: en.m.k };
    }
    if (this.plan.name === "bot") return input;
    const out = this.policy(this.plan.move, base, p.x, p.y, p.angle, en.m.x, en.m.y, en.m.r, true);
    out.weapon = input.weapon || 0;
    return out;
  }
}
