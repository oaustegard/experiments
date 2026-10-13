#include "lstate.h"
#include "fn_luaH_getshortstr.h"
#include "fn_luaT_callTMres.h"
#include "fn_luaT_gettmbyobj.h"

static int callbinTM (lua_State *L, const TValue *p1, const TValue *p2,
                      StkId res, TMS event) {
  const TValue *tm = luaT_gettmbyobj(L, p1, event);  /* try first operand */
  if ((tm->tt_ & 0xf) == 0)
    tm = luaT_gettmbyobj(L, p2, event);  /* try second operand */
  if ((tm->tt_ & 0xf) == 0) return 0;
  luaT_callTMres(L, tm, p1, p2, res);
  return 1;
}
