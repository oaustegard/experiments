#include "lstate.h"
#include "LClosure.h"
#include "fn_luaU_dump.h"
#include "lua.h"

LUA_API int lua_dump (lua_State *L, lua_Writer writer, void *data, int strip) {
  int status;
  TValue *o;
  o = L->top - 1;
  if (o->tt_ == (6 | (0 << 4) | (1 << 6))) {
    LClosure *f = (LClosure *)o->value_.gc;
    status = luaU_dump(L, f->p, writer, data, strip);
  }
  else
    status = 1;
  return status;
}
