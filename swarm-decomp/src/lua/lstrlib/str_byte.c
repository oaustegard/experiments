static int str_byte (lua_State *L) {
  size_t l;
  const char *s = luaL_checklstring(L, 1, &l);
  lua_Integer pi = luaL_optinteger(L, 2, 1);
  size_t posi = posrelatI(pi, l);
  size_t pose = getendpos(L, 3, pi, l);
  int n, i;
  if (posi > pose) return 0;
  if (__builtin_expect(((pose - posi) >= (size_t)2147483647), 0))
    return luaL_error(L, "string slice too long");
  n = (int)(pose - posi) + 1;
  luaL_checkstack(L, n, "string slice too long");
  for (i = 0; i < n; i++)
    lua_pushinteger(L, (unsigned char)(s[posi + i - 1]));
  return n;
}
