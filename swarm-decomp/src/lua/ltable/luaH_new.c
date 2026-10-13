#include "fn_luaC_newobj.h"

Table *luaH_new(lua_State *L) {
  GCObject *o = luaC_newobj(L, LUA_TTABLE, sizeof(Table));
  Table *t = (Table *)o;
  t->metatable = NULL;
  t->array = NULL;
  t->alimit = 0;
  t->node = (Node *)&dummynode_;
  t->flags = 0x3f;
  t->lsizenode = 0;
  t->lastfree = NULL;
  return t;
}
