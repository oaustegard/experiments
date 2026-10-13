static int str_packsize (lua_State *L) {
  Header h;
  const char *fmt = luaL_checklstring(L, 1, NULL);
  size_t totalsize = 0;
  h.L = L;
  h.islittle = 1;
  h.maxalign = 1;
  while (*fmt != '\0') {
    int size, ntoalign;
    KOption opt = getdetails(&h, totalsize, &fmt, &size, &ntoalign);
    luaL_argcheck(L, opt != Kstring && opt != Kzstr, 1,
                     "variable-length format");
    size += ntoalign;
    luaL_argcheck(L, totalsize <= (size_t)INT_MAX - size, 1,
                     "format result too large");
    totalsize += size;
  }
  lua_pushinteger(L, (lua_Integer)totalsize);
  return 1;
}
