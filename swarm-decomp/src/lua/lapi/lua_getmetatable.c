#include "lstate.h"
#include "lua.h"

LUA_API int lua_getmetatable (lua_State *L, int objindex) {
  const TValue *obj;
  Table *mt;
  int res = 0;
  obj = index2value(L, objindex);
  switch (obj->tt_ & 0xf) {
    case 5: mt = ((Table *)obj->value_.gc)->metatable; break;
    case 7: mt = ((Udata *)obj->value_.gc)->metatable; break;
    default: mt = L->l_G->mt[obj->tt_ & 0xf]; break;
  }
  if (mt != NULL) {
    L->top->value_.gc = (GCObject *)mt;
    L->top->tt_ = 0x45;
    L->top++;
    res = 1;
  }
  return res;
}
