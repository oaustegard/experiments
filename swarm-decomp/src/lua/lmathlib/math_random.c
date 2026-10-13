static int math_random (lua_State *L) {
  lua_Integer low, up;
  lua_Unsigned p;
  RanState *g = (RanState *)lua_touserdata(L, lua_upvalueindex(1));
  Rand64 rv = nextrand(g->s);  /* next pseudo-random value */
  switch (lua_gettop(L)) {  /* check number of arguments */
    case 0: {  /* no arguments */
      lua_pushnumber(L, (lua_Number)(rv >> 11) * 0x1.0p-53);
      return 1;
    }
    case 1: {  /* only upper limit */
      low = 1;
      up = luaL_checkinteger(L, 1);
      if (up == 0) {  /* single 0 as argument? */
        lua_pushinteger(L, (lua_Integer)rv);  /* full random integer */
        return 1;
      }
      break;
    }
    case 2: {  /* lower and upper limits */
      low = luaL_checkinteger(L, 1);
      up = luaL_checkinteger(L, 2);
      break;
    }
    default: return luaL_error(L, "wrong number of arguments");
  }
  /* random integer in the interval [low, up] */
  luaL_argcheck(L, low <= up, 1, "interval is empty");
  /* project random integer into the interval [0, up - low] */
  p = project((lua_Unsigned)rv, (lua_Unsigned)up - (lua_Unsigned)low, g);
  lua_pushinteger(L, p + (lua_Unsigned)low);
  return 1;
}
