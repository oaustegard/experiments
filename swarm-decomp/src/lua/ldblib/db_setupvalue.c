#include <lua.h>
#include <lauxlib.h>

static int auxupvalue (lua_State *L, int get);

static int db_setupvalue (lua_State *L) {
  luaL_checkany(L, 3);
  return auxupvalue(L, 0);
}
