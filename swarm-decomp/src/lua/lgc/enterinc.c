static void enterinc(global_State *g) {
  whitelist(g, g->allgc);
  g->survival = g->old1 = g->reallyold = NULL;
  whitelist(g, g->finobj);
  whitelist(g, g->tobefnz);
  g->finobjsur = g->finobjold1 = g->finobjrold = NULL;
  g->gcstate = GCSpause;
  g->gckind = KGC_INC;
  g->lastatomic = 0;
}
