#include <stddef.h>
#include "lstate.h"
#include "fn_luaO_tostring.h"
#include "fn_luaC_step.h"
#include "TString.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

LUA_API const char *lua_tolstring (lua_State *L, int idx, size_t *len) {
  TValue *o;
  o = index2value(L, idx);
  if ((o->tt_ & 0xf) != 4) {
    if ((o->tt_ & 0xf) != 3) {
      if (len != NULL)
        *len = 0;
      return NULL;
    }
    luaO_tostring(L, o);
    if (L->l_G->GCdebt > 0)
      luaC_step(L);
    o = index2value(L, idx);
  }
  if (len != NULL) {
    TString *ts = (TString *)o->value_.gc;
    *len = (ts->shrlen != 0xff) ? ts->shrlen : ts->u.lnglen;
  }
  return ((TString *)o->value_.gc)->contents;
}
