static int add_value (MatchState *ms, luaL_Buffer *b, const char *s, const char *e, int tr) {
  lua_State *L = ms->L;
  switch (tr) {
    case LUA_TFUNCTION: {
      int n;
      lua_pushvalue(L, 3);
      n = push_captures(ms, s, e);
      lua_callk(L, n, 1, 0, NULL);
      break;
    }
    case LUA_TTABLE: {
      push_onecapture(ms, 0, s, e);
      lua_gettable(L, 3);
      break;
    }
    default: {
      add_s(ms, b, s, e);
      return 1;
    }
  }
  if (!lua_toboolean(L, -1)) {
    lua_pop(L, 1);
    luaL_addlstring(b, s, e - s);
    return 0;
  }
  else if (luai_unlikely(!lua_isstring(L, -1)))
    return luaL_error(L, "invalid replacement value (a %s)", luaL_typename(L, -1));
  else {
    luaL_addvalue(b);
    return 1;
  }
}
