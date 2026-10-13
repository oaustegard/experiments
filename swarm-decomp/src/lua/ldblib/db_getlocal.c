#include <lua.h>
#include <lauxlib.h>

static lua_State *getthread (lua_State *L, int *arg);
static void checkstack (lua_State *L, lua_State *L1, int n);

static int db_getlocal (lua_State *L) {
  int arg;
  lua_State *L1 = getthread(L, &arg);
  int nvar = (int)luaL_checkinteger(L, arg + 2);
  if (lua_isfunction(L, arg + 1)) {
    lua_pushvalue(L, arg + 1);
    lua_pushstring(L, lua_getlocal(L, NULL, nvar));
    return 1;
  }
  else {
    lua_Debug ar;
    const char *name;
    int level = (int)luaL_checkinteger(L, arg + 1);
    if (luai_unlikely(!lua_getstack(L1, level, &ar)))
      return luaL_argerror(L, arg+1, "level out of range");
    checkstack(L, L1, 1);
    name = lua_getlocal(L1, &ar, nvar);
    if (name) {
      lua_xmove(L1, L, 1);
      lua_pushstring(L, name);
      lua_rotate(L, -2, 1);
      return 2;
    }
    else {
      lua_pushnil(L);
      return 1;
    }
  }
}
