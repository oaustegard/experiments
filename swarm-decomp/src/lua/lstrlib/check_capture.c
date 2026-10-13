static int check_capture (MatchState *ms, int l) {
  l -= '1';
  if (__builtin_expect(((l < 0 || l >= ms->level || ms->capture[l].len == CAP_UNFINISHED) != 0), 0))
    return luaL_error(ms->L, "invalid capture index %%%d", l + 1);
  return l;
}
