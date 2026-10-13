static int macroarg(struct sbuf *b, int term, int (*rd)(void), void (*put)(int)) {
  int c, q, sp;
  c = rd();
  while (c == ' ') c = rd();
  q = (c == '\n') | (c == term);
  if (q) {
    put(c);
    return 1;
  }
  if (c < 0) return 1;
  sp = 1;
  if (c == '"') {
    c = rd();
    q = (c >= 0) & (c != '\n');
    if (!q) goto fin;
    sp = 0;
  }
  for (;;) {
    if (c == '"' && q) {
      c = rd();
      if (c != '"') goto fin;
    }
    if (c == c_ec) {
      sbuf_add(b, c);
      c = rd();
    }
    sbuf_add(b, c);
    c = rd();
    if (c < 0 || c == '\n') goto fin;
    if (c == term && !q) goto endput;
    if (c == ' ' && sp) goto endput;
  }
endput:
  sbuf_add(b, 0);
  put(c);
  return 0;
fin:
  sbuf_add(b, 0);
  if (c >= 0) put(c);
  return 0;
}
