static int str_gsub (lua_State *L) {
  size_t srcl, lp;
  const char *src = luaL_checklstring(L, 1, &srcl);
  const char *p = luaL_checklstring(L, 2, &lp);
  const char *lastmatch = NULL;
  int tr = lua_type(L, 3);
  lua_Integer max_s = luaL_optinteger(L, 4, srcl + 1);
  int anchor = (*p == '^');
  lua_Integer n = 0;
  int changed = 0;
  MatchState ms;
  luaL_Buffer b;
  luaL_argexpected(L, tr == LUA_TNUMBER || tr == LUA_TSTRING ||
                   tr == LUA_TFUNCTION || tr == LUA_TTABLE, 3,
                   "string/function/table");
  luaL_buffinit(L, &b);
  if (anchor) {
    p++; lp--;
  }
  prepstate(&ms, L, src, srcl, p, lp);
  while (n < max_s) {
    const char *e;
    reprepstate(&ms);
    if ((e = match(&ms, src, p)) != NULL && e != lastmatch) {
      n++;
      changed = add_value(&ms, &b, src, e, tr) | changed;
      src = lastmatch = e;
    }
    else if (src < ms.src_end)
      luaL_addchar(&b, *src++);
    else break;
    if (anchor) break;
  }
  if (!changed)
    lua_pushvalue(L, 1);
  else {
    luaL_addlstring(&b, src, ms.src_end - src);
    luaL_pushresult(&b);
  }
  lua_pushinteger(L, n);
  return 2;
}
