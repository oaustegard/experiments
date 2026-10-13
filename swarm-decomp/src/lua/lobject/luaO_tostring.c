#include "tostringbuff.h"
#include "fn_luaS_newlstr.h"

void luaO_tostring (lua_State *L, TValue *obj) {
  char buff[44];
  int len = tostringbuff(obj, buff);
  GCObject *ts = (GCObject *)luaS_newlstr(L, buff, len);
  obj->value_.gc = ts;
  obj->tt_ = ts->tt | 0x40;
}
