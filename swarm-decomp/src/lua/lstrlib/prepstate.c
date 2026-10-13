static void prepstate (MatchState *ms, lua_State *L, const char *s, size_t ls, const char *p, size_t lp) {
  ms->L = L;
  ms->matchdepth = 200;
  ms->src_init = s;
  ms->src_end = s + ls;
  ms->p_end = p + lp;
}
