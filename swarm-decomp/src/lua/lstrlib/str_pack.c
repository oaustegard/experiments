static int str_pack (lua_State *L) {
  luaL_Buffer b;
  Header h;
  const char *fmt = luaL_checkstring(L, 1);
  int arg = 1;
  size_t totalsize = 0;
  h.L = L;
  h.islittle = nativeendian.little;
  h.maxalign = 1;
  lua_pushnil(L);
  luaL_buffinit(L, &b);
  while (*fmt != '\0') {
    int size, ntoalign;
    KOption opt = getdetails(&h, totalsize, &fmt, &size, &ntoalign);
    totalsize += ntoalign + size;
    while (ntoalign-- > 0)
      luaL_addchar(&b, LUAL_PACKPADBYTE);
    arg++;
    switch (opt) {
      case Kint: {
        lua_Integer n = luaL_checkinteger(L, arg);
        if (size < SZINT) {
          lua_Integer lim = (lua_Integer)1 << ((size * NB) - 1);
          luaL_argcheck(L, -lim <= n && n < lim, arg, "integer overflow");
        }
        packint(&b, (lua_Unsigned)n, h.islittle, size, (n < 0));
        break;
      }
      case Kuint: {
        lua_Integer n = luaL_checkinteger(L, arg);
        if (size < SZINT)
          luaL_argcheck(L, (lua_Unsigned)n < ((lua_Unsigned)1 << (size * NB)),
                           arg, "unsigned overflow");
        packint(&b, (lua_Unsigned)n, h.islittle, size, 0);
        break;
      }
      case Kfloat: {
        float f = (float)luaL_checknumber(L, arg);
        char *buff = luaL_prepbuffsize(&b, sizeof(f));
        copywithendian(buff, (char *)&f, sizeof(f), h.islittle);
        luaL_addsize(&b, size);
        break;
      }
      case Knumber: {
        double f = (double)luaL_checknumber(L, arg);
        char *buff = luaL_prepbuffsize(&b, sizeof(f));
        copywithendian(buff, (char *)&f, sizeof(f), h.islittle);
        luaL_addsize(&b, size);
        break;
      }
      case Kdouble: {
        double f = (double)luaL_checknumber(L, arg);
        char *buff = luaL_prepbuffsize(&b, sizeof(f));
        copywithendian(buff, (char *)&f, sizeof(f), h.islittle);
        luaL_addsize(&b, size);
        break;
      }
      case Kchar: {
        size_t len;
        const char *s = luaL_checklstring(L, arg, &len);
        luaL_argcheck(L, len <= (size_t)size, arg, "string longer than given size");
        luaL_addlstring(&b, s, len);
        while (len++ < (size_t)size)
          luaL_addchar(&b, LUAL_PACKPADBYTE);
        break;
      }
      case Kstring: {
        size_t len;
        const char *s = luaL_checklstring(L, arg, &len);
        luaL_argcheck(L, size >= (int)sizeof(size_t) ||
                         len < ((size_t)1 << (size * NB)),
                         arg, "string length does not fit in given size");
        packint(&b, (lua_Unsigned)len, h.islittle, size, 0);
        luaL_addlstring(&b, s, len);
        totalsize += len;
        break;
      }
      case Kzstr: {
        size_t len;
        const char *s = luaL_checklstring(L, arg, &len);
        luaL_argcheck(L, strlen(s) == len, arg, "string contains zeros");
        luaL_addlstring(&b, s, len);
        luaL_addchar(&b, '\0');
        totalsize += len + 1;
        break;
      }
      case Kpadding: luaL_addchar(&b, LUAL_PACKPADBYTE);  /* FALLTHROUGH */
      case Kpaddalign: case Knop:
        arg--;
        break;
    }
  }
  luaL_pushresult(&b);
  return 1;
}
