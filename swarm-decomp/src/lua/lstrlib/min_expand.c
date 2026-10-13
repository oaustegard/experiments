static const char *min_expand (MatchState *ms, const char *s, const char *p, const char *ep) {
  for (;;) {
    const char *res = match(ms, s, ep+1);
    if (res != NULL)
      return res;
    else if (singlematch(ms, s, p, ep))
      s++;
    else return NULL;
  }
}
