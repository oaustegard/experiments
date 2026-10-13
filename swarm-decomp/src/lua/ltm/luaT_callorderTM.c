#include "lstate.h"
#include "lobject_tags.h"
#include "fn_luaG_errors.h"
#include "fn_luaT_callTMres.h"
#include "fn_luaT_gettmbyobj.h"

int luaT_callorderTM (lua_State *L, const TValue *p1, const TValue *p2,
                      TMS event) {
  if (callbinTM(L, p1, p2, L->top, event))  /* try original event */
    return !l_isfalse(L->top);
  else if (event == TM_LE) {
    /* try '!(p2 < p1)' for '(p1 <= p2)' */
    L->ci->callstatus |= (1 << 13);  /* mark it is doing 'lt' for 'le' */
    if (callbinTM(L, p2, p1, L->top, TM_LT)) {
      L->ci->callstatus ^= (1 << 13);  /* clear mark */
      return l_isfalse(L->top);
    }
  }
  luaG_ordererror(L, p1, p2);  /* no metamethod found */
  return 0;  /* to avoid warnings */
}
