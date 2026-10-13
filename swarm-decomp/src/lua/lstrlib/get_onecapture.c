static size_t get_onecapture (MatchState *ms, int i, const char *s, const char *e, const char **cap) {
  if (i >= ms->level) {
    if (__builtin_expect(((i != 0) != 0), 0))
      luaL_error(ms->L, "invalid capture index %%%d", i + 1);
    *cap = s;
    return e - s;
  }
  else {
    ptrdiff_t capl = ms->capture[i].len;
    *cap = ms->capture[i].init;
    if (__builtin_expect(((capl == CAP_UNFINISHED) != 0), 0))
      luaL_error(ms->L, "unfinished capture");
    else if (capl == CAP_POSITION)
      lua_pushinteger(ms->L, (ms->capture[i].init - ms->src_init) + 1);
    return capl;
  }
}
