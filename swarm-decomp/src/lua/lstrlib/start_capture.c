static const char *start_capture (MatchState *ms, const char *s, const char *p, int what) {
  const char *res;
  int level = ms->level;
  if (level >= 32)
    luaL_error(ms->L, "too many captures");
  ms->capture[level].init = s;
  ms->capture[level].len = what;
  ms->level = level+1;
  if ((res=match(ms, s, p)) == NULL)
    ms->level--;
  return res;
}
