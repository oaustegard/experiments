static int tinsert (lua_State *L) {
  lua_Integer pos;
  lua_Integer e = aux_getn(L, 1, TAB_RW);
  e = luaL_intop(+, e, 1);
  switch (lua_gettop(L)) {
    case 2: {
      pos = e;
      break;
    }
    case 3: {
      lua_Integer i;
      pos = luaL_checkinteger(L, 2);
      luaL_argcheck(L, (lua_Unsigned)pos - 1u < (lua_Unsigned)e, 2,
                       "position out of bounds");
      for (i = e; i > pos; i--) {
        lua_geti(L, 1, i - 1);
        lua_seti(L, 1, i);
      }
      break;
    }
    default: {
      return luaL_error(L, "wrong number of arguments to 'insert'");
    }
  }
  lua_seti(L, 1, pos);
  return 0;
}
