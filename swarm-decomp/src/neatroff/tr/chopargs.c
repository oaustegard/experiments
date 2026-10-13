static void chopargs(struct sbuf *b, char **args) {
  char *p = sbuf_buf(b);
  char *end = p + sbuf_len(b);
  int i = 1;
  int more = 0;
  if (p == 0 || p >= end) return;
  do {
    args[i - 1] = p;
    p = memchr(p, 0, end - p);
    if (p != 0) {
      p++;
      more = (p < end) & (i <= 31);
      i++;
    } else
      more = 0;
  } while (more);
}
