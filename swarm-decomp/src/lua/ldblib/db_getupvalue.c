#include <lua.h>
#include <lauxlib.h>

static int auxupvalue (lua_State *L, int get);

static int db_getupvalue (lua_State *L) {
  return auxupvalue(L, 1);
}
