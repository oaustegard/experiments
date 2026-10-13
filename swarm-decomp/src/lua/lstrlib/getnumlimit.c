static int getnumlimit (Header *h, const char **fmt, int df) {
  int sz = getnum(fmt, df);
  if (__builtin_expect(((sz > 16 || sz <= 0) != 0), 0))
    return luaL_error(h->L, "integral size (%d) out of limits [1,%d]", sz, 16);
  return sz;
}
