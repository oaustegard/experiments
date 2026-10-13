static int quotefloat (lua_State *L, char *buff, lua_Number n) {
  const char *s;
  if (n == (lua_Number)HUGE_VAL)
    s = "1e9999";
  else if (n == -(lua_Number)HUGE_VAL)
    s = "-1e9999";
  else if (n != n)
    s = "(0/0)";
  else {
    int nb = snprintf(buff, MAX_ITEM, "%a", n);
    if (memchr(buff, '.', nb) == NULL) {
      char point = *localeconv()->decimal_point;
      char *ppoint = (char *)memchr(buff, point, nb);
      if (ppoint) *ppoint = '.';
    }
    return nb;
  }
  return snprintf(buff, MAX_ITEM, "%s", s);
}
