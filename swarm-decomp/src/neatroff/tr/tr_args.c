void tr_args(struct tr_args *a, int term, int (*rd)(void), void (*put)(int)) {
  struct sbuf b;
  sbuf_init(&b);
  while (!macroarg(&b, term, rd, put))
    ;
  chopargs(&b, (char **)a);
  sbuf_out(&b);
}
