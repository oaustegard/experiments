// Debug-only planner: exact BFS labels, the targets TRM is trained on. Used to test the bot's
// controller without a GPU; never used for the reported runs.
import { shortestUnion } from "./grid.js?v=20261030-123935";
export class OraclePlanner { async plan(g) { return { path: shortestUnion(g, Math.round(Math.sqrt(g.length))), steps: 0, ms: 0 }; } }
