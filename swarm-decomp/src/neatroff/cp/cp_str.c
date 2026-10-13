/* cp_str: expand a copy-mode string or request name and push it for reading */
static void cp_str(void)
{
  char buf[128];
  char *args[34] = {0};
  char *p = 0;

  args[0] = buf;
  if (cparg(buf, cp_next)) {
    p = tr_args(&args[1], ']', cp_noninext, in_back);
    while (cp_next() == 4)
      ;
  }
  if (str_get(map(buf)))
    in_push(str_get(map(buf)), args);
  else if (*nreg('C') == 0)
    tr_req(map(buf), args);
  free(p);
}
