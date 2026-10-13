#include "lstate.h"
#include "lua.h"
#include "Table.h"

static void auxsetstr (lua_State *L, const TValue *t, const char *k);

LUA_API void lua_setglobal (lua_State *L, const char *name) {
  const TValue *G = &((Table *)L->l_G->l_registry.value_.gc)->array[1];
  auxsetstr(L, G, name);
}
