#include "lstate.h"
#include "StackValue.h"
#include "lobject_setters.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaD_seterrorobj.h"

void luaD_seterrorobj (lua_State *L, int errcode, StkId oldtop) {
  StackValue *so = (StackValue *)oldtop;
  switch (errcode) {
    case LUA_ERRMEM: {
      TString *ts = G(L)->memerrmsg;
      so->val.value_.gc = (GCObject *)ts;
      so->val.tt_ = ts->tt | BIT_ISCOLLECTABLE;
      break;
    }
    case LUA_ERRERR: {
      TString *ts = luaS_newlstr(L, "error in error handling", 23);
      so->val.value_.gc = (GCObject *)ts;
      so->val.tt_ = ts->tt | BIT_ISCOLLECTABLE;
      break;
    }
    case LUA_OK: {
      setnilvalue(&so->val);
      break;
    }
    default: {
      setobjs2s(L, &so->val, L->top - 1);
      break;
    }
  }
  L->top = (StkId)(so + 1);
}
