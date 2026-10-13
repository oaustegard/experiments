#include "BuffFS.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaV_concat.h"

static void pushstr (BuffFS *buff, const char *str, size_t lstr) {
  lua_State *L = buff->L;
  StkId top = L->top;
  GCObject *ts = (GCObject *)luaS_newlstr(L, str, lstr);
  top->value_.gc = ts;
  top->tt_ = ts->tt | 0x40;
  L->top++;
  if (!buff->pushed)
    buff->pushed = 1;
  else
    luaV_concat(L, 2);
}
