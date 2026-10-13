#define l_likely(x)	__builtin_expect(((x) != 0), 1)
static void trymt (lua_State *L, const char *mtname) {
  lua_settop(L, 2);
  if (l_likely(lua_type(L, 2) == LUA_TSTRING || !luaL_getmetafield(L, 2, mtname)))
    luaL_error(L, "attempt to %s a '%s' with a '%s'", mtname + 2,
                  luaL_typename(L, -2), luaL_typename(L, -1));
  lua_rotate(L, -3, 1);
  lua_callk(L, 2, 1, 0, NULL);
}

#undef l_likely
