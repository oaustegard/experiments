#include "lstate.h"
#include "cclosure.h"
#include "lua.h"
#include "fn_luaC_barrier_.h"
#include "fn_luaC_checkfinalizer.h"
static TValue *index2value (lua_State *L, int idx);
LUA_API int lua_setmetatable (lua_State *L, int objindex) {
  TValue *obj;
  Table *mt;
  obj = index2value(L, objindex);
  if ((L->top[-1].tt_ & 0xf) == 0)
    mt = NULL;
  else
    mt = (Table *)L->top[-1].value_.gc;
  switch (obj->tt_ & 0xf) {
    case 5: {
      ((Table *)obj->value_.gc)->metatable = mt;
      if (mt) {
        if ((((Table *)obj->value_.gc)->marked & 0x20) && (mt->marked & 0x18))
          luaC_barrier_(L, (GCObject *)obj->value_.gc, (GCObject *)mt);
        luaC_checkfinalizer(L, (GCObject *)obj->value_.gc, mt);
      }
      break;
    }
    case 7: {
      ((Udata *)obj->value_.gc)->metatable = mt;
      if (mt) {
        if ((((Udata *)obj->value_.gc)->marked & 0x20) && (mt->marked & 0x18))
          luaC_barrier_(L, (GCObject *)obj->value_.gc, (GCObject *)mt);
        luaC_checkfinalizer(L, (GCObject *)obj->value_.gc, mt);
      }
      break;
    }
    default: {
      L->l_G->mt[obj->tt_ & 0xf] = mt;
      break;
    }
  }
  L->top--;
  return 1;
}
