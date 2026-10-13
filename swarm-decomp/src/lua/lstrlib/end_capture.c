static const char *end_capture (MatchState *ms, const char *s, const char *p) {
  int l = capture_to_close(ms);
  const char *res;
  ms->capture[l].len = s - ms->capture[l].init;
  if ((res = match(ms, s, p)) == NULL)
    ms->capture[l].len = CAP_UNFINISHED;
  return res;
}
