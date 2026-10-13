/* cp_args: join the arguments of the current macro into one string and push it */
static void cp_args(int quote, int bq)
{
  struct sbuf b;
  char *q, *sep, *s;
  int i;

  sbuf_init(&b);
  q = (quote & 1) == 0 ? "" : "\"";
  if (in_nargs() > 1) {
    sep = "";
    i = 1;
    do {
      sbuf_append(&b, sep);
      sbuf_append(&b, q);
      s = in_arg(i);
      while (*s) {
        sbuf_append(&b, (*s == '"' && (bq & 1)) ? "\"" : "");
        sbuf_add(&b, (unsigned char)*s);
        s++;
      }
      sbuf_append(&b, q);
      i++;
      sep = " ";
    } while (in_nargs() > i);
  }
  in_push(sbuf_buf(&b), 0);
  sbuf_done(&b);
}
