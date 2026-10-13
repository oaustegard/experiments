#include "lstate.h"
#include "lua.h"

LUA_API int lua_compare (lua_State *L, int index1, int index2, int op) {
  const TValue *o1;
  const TValue *o2;
  int i = 0;
  o1 = index2value(L, index1);
  o2 = index2value(L, index2);
  if ((((o1->tt_ & 0xf) != 0) || o1 != &L->l_G->nilvalue) &&
      (((o2->tt_ & 0xf) != 0) || o2 != &L->l_G->nilvalue)) {
    switch (op) {
      case LUA_OPEQ: i = luaV_equalobj(L, o1, o2); break;
      case LUA_OPLT: i = luaV_lessthan(L, o1, o2); break;
      case LUA_OPLE: i = luaV_lessequal(L, o1, o2); break;
      default: break;
    }
  }
  return i;
}
