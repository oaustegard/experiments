int tr_nextreq(void) {
  int c, n, cnt;
  size_t len;
  char *buf, *name;
  if (!tr_nl) return 1;
  c = cp_next();
  if (c == c_ec) goto esc;
main:
  if (c < 0) goto back;
  if (c == c_cc || c == c_c2) goto ctrl;
  if (c == '\n') goto nl;
  if (c != ' ') goto back;
  if (tr_sm < 0) goto back;
  cp_reqbeg();
sm2:
  cnt = 0;
  buf = xmalloc(strlen(map_name(tr_sm)) + 1);
  strcpy(buf, map_name(tr_sm));
  do {
    c = cp_next();
    cnt++;
  } while (c == ' ');
  in_back(c);
  *nreg(map("lsn")) = cnt;
  len = strlen(buf);
tail:
  name = xmalloc(len + 2);
  name[0] = '.';
  strcpy(name + 1, buf);
  tr_nextreq_exec(buf, name, 0);
  free(name);
  free(buf);
  return 0;
nl:
  if (tr_bm < 0) goto back;
  cp_reqbeg();
bm:
  buf = xmalloc(strlen(map_name(tr_bm)) + 1);
  len = stpcpy(buf, map_name(tr_bm)) - buf;
  goto tail;
back:
  in_back(c);
  return 1;
ctrl:
  cp_reqbeg();
  if (c == '\n') goto bm;
  if (c == ' ') goto sm2;
  name = read_name(*nreg(0x43));
  buf = xmalloc(strlen(name) + 2);
  buf[0] = c;
  strcpy(buf + 1, name);
  tr_nextreq_exec(name, buf, 1);
  free(buf);
  free(name);
  return 0;
esc:
  n = cp_next();
  if (n == '!') {
    char *a[35] = {0};
    struct sbuf b;
    a[0] = "\\!";
    sbuf_init(&b);
    cp_copymode(1);
    mkargs_eol(&b);
    cp_copymode(0);
    chopargs(&b, a + 1);
    tr_transparent((struct tr_args *)a);
    sbuf_done(&b);
    return 0;
  }
  in_back(n);
  goto main;
}
