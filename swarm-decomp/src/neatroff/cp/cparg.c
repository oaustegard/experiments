/* cparg: parse one argument of a copy-mode escape into p; returns 1 if it ended at a blank */
static int cparg(char *p, int (*next)(void))
{
  int c, n, m;

  c = next();
  while (c == 4)
    c = next();
  if (c == '(') {
    n = utf8next(p, cp_noninext);
    utf8next(p + n, cp_noninext);
    return 0;
  }
  if (*nreg('C') == 0 && c == '[') {
    do
      c = next();
    while (c == 4);
    if (c < 0 || c == ']') {
      *p = 0;
      return 0;
    }
    if (c == ' ') {
      *p = 0;
      return 1;
    }
    n = 0;
    do {
      if ((m = n + 1) <= 0x7f) {
        p[n] = c;
        n = m;
      }
      do
        c = next();
      while (c == 4);
      if (c < 0 || c == ']') {
        p += n;
        *p = 0;
        return 0;
      }
    } while (c != ' ');
    p += n;
    *p = 0;
    return 1;
  }
  in_back(c);
  utf8next(p, cp_noninext);
  return 0;
}
