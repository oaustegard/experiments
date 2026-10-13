/* cp_next: next character of copy mode, expanding the escapes that copy mode interprets */
int cp_next(void)
{
  int c, ev;
  char *orig, *s, *st, *end;
  char buf[128];
  char cbuf[4];

  for (;;) {
    if (in_top() >= 0)
      return in_next();
    c = cp_raw();
    if (c != c_ec)
      return c;
    c = cp_raw();
  dispatch:
    switch (c) {
    case 'E':
      if (cp_cpmode) {
        in_back(c);
        return c_ec;
      }
      c = cp_next();
      goto dispatch;
    case '"':
      while ((c = cp_raw()) >= 0 && c != '\n')
        ;
      return c;
    case '#':
      do {
        c = cp_raw();
        if (c < 0)
          return c;
      } while (c != '\n');
      continue;
    case 'w':
      if (cp_cpmode)
        goto defcase;
      sprintf(buf, "%d", ren_wid(cp_next, in_back));
      in_push(buf, 0);
      continue;
    case '*':
      cp_str();
      continue;
    case 'R':
      if (cp_cpmode)
        goto defcase;
      orig = quotednext(cp_noninext, in_back);
      for (s = orig; *s & 0xdf; s++)
        ;
      if (*s) {
        *s = 0;
        num_set(map(orig), eval_re(s + 1, *nreg(map(orig)), 'u'));
      }
      free(orig);
      continue;
    case '?':
      orig = quotednext(cp_noninext, in_back);
      s = orig;
      ev = eval_up(&s, 0);
      if (charread(&s, cbuf) < 0)
        goto done;
      if (cbuf[0] == '\\' && cbuf[1] == '&') {
        if (!cbuf[2] && charread(&s, cbuf) < 0)
          goto done;
      }
      st = s;
      end = st;
      while (charread_delim(&s, buf, cbuf) >= 0)
        end = s;
      *end = 0;
      end = s;
      while (charread_delim(&s, buf, cbuf) >= 0)
        end = s;
      *end = 0;
      if (ev > 0)
        s = st;
      in_push(s, 0);
    done:
      free(orig);
      continue;
    case '$':
      if (cp_cpmode)
        goto defcase;
      cparg(buf, cp_next);
      if (buf[0] == '@' && buf[1] == 0) {
        cp_args(1, 0);
        continue;
      }
      if (buf[0] == '*' && buf[1] == 0) {
        cp_args(0, 0);
        continue;
      }
      if (buf[0] == '^' && buf[1] == 0) {
        cp_args(1, 1);
        continue;
      }
      {
        unsigned n = strtol(buf, 0, 10);
        if (n > 0x1f)
          continue;
        s = in_arg(n);
        if (s)
          in_push(s, 0);
      }
      continue;
    case 'g':
      if (cp_cpmode)
        goto defcase;
      cparg(buf, cp_next);
      in_push(num_getfmt(map(buf)), 0);
      continue;
    case 'n':
      do
        c = cp_next();
      while (c == 4);
      if (c == '+' || c == '-') {
        cparg(buf, cp_next);
        ev = map(buf);
        num_inc(ev, c == '+');
      } else {
        in_back(c);
        cparg(buf, cp_next);
        ev = map(buf);
      }
      s = num_str(ev);
      if (s)
        in_push(s, 0);
      continue;
    default:
    defcase:
      in_back(c);
      return c_ec;
    }
  }
}
