static void rname (const Proto *p, int pc, int c, const char **name) {
  const char *what = basicgetobjname(p, &pc, c, name);
  if (!(what && *what == 'c'))
    *name = "?";
}
