#include "lstate.h"
#include "lua.h"
#include "Table.h"

LUA_API void lua_setfield (lua_State *L, int idx, const char *k) {
  auxsetstr(L, index2value(L, idx), k);
}
