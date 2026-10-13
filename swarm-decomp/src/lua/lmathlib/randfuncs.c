static int math_random (lua_State *L);
static int math_randomseed (lua_State *L);

static const luaL_Reg randfuncs[] = {
  {"random", math_random},
  {"randomseed", math_randomseed},
  {NULL, NULL}
};
