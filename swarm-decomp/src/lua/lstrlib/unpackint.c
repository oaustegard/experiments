static lua_Integer unpackint (lua_State *L, const char *str,
                              int islittle, int size, int issigned) {
  lua_Unsigned res = 0;
  int i;
  int limit = (size <= 8) ? size : 8;
  for (i = limit - 1; i >= 0; i--) {
    res <<= 8;
    res |= (lua_Unsigned)(unsigned char)str[islittle ? i : size - 1 - i];
  }
  if (size < 8) {
    if (issigned) {
      lua_Unsigned mask = (lua_Unsigned)1 << (size * 8 - 1);
      res = ((res ^ mask) - mask);
    }
  }
  else if (size > 8) {
    int mask = (!issigned || (lua_Integer)res >= 0) ? 0 : 0xff;
    for (i = limit; i < size; i++) {
      if (__builtin_expect(((unsigned char)str[islittle ? i : size - 1 - i] != mask), 0))
        luaL_error(L, "%d-byte integer does not fit into Lua Integer", size);
    }
  }
  return (lua_Integer)res;
}
