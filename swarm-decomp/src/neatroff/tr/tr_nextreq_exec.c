static void tr_nextreq_exec(char *name, char *arg0, int flag) {
  char *a[35] = {0};
  struct sbuf b;
  struct trcmd *p;
  char *s;
  int c;
  a[0] = arg0;
  p = (struct trcmd *)str_dget(map(name));
  s = str_get(map(name));
  sbuf_init(&b);
  if (flag) {
    if (p) {
      void (*mk)(struct sbuf *) = (void (*)(struct sbuf *))p->mk;
      if (!mk)
        mkargs_req(&b);
      else {
        mk(&b);
        if (!p->mk) mkargs_req(&b);
      }
    } else {
      cp_copymode(1);
      while (!macroarg(&b, -1, cp_next, in_back))
        continue;
      do {
        c = cp_next();
      } while (c >= 0 && c != '\n');
      cp_copymode(0);
    }
    chopargs(&b, a + 1);
    if (s)
      in_push(s, a);
    else if (p)
      ((void (*)(char **))p->fn)(a);
    sbuf_done(&b);
  } else {
    if (s)
      in_push(s, a);
    else if (p)
      ((void (*)(char **))p->fn)(a);
    sbuf_done(&b);
  }
}
