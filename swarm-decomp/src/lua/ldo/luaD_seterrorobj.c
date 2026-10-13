#include "lstate.h"
#include "lobject_setters.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaD_seterrorobj.h"

void luaD_seterrorobj (lua_State *L, int errcode, StkId oldtop) {
  StkId top = oldtop + 1;
  switch (errcode) {
    case LUA_ERRMEM: {
      TString *ts = G(L)->memerrmsg;
      top[-1].value_.gc = (GCObject *)ts;
      top[-1].tt_ = ts->tt | BIT_ISCOLLECTABLE;
      break;
    }
    case LUA_ERRERR: {
      TString *ts = luaS_newlstr(L, "error in error handling", 23);
      top[-1].value_.gc = (GCObject *)ts;
      top[-1].tt_ = ts->tt | BIT_ISCOLLECTABLE;
      break;
    }
    case LUA_OK: {
      setnilvalue(&top[-1]);
      break;
    }
    default: {
      setobjs2s(L, &top[-1], L->top - 1);
      break;
    }
  }
  L->top = top;
}
