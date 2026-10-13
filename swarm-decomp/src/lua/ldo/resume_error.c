#include "lstate.h"
#include "fn_luaS_new.h"

static int resume_error (lua_State *L, const char *msg, int narg) {
  StkId top = L->top - narg;
  struct TString *s;
  L->top = top;
  s = luaS_new(L, msg);
  top->value_.gc = (GCObject *)s;
  top->tt_ = ((GCObject *)s)->tt | 0x40;
  L->top++;
  return 2;
}
