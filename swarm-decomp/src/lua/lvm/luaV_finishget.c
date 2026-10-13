#include "lstate.h"
#include "ltags.h"
#include "Table.h"
#include "TString.h"
#include "fn_luaT_gettmbyobj.h"
#include "fn_luaT_gettm.h"
#include "fn_luaT_callTMres.h"
#include "fn_luaH_get.h"
#include "fn_luaG_typeerror.h"
#include "fn_luaG_runerror.h"

#define MAXTAGLOOP 2000

void luaV_finishget (lua_State *L, const TValue *t, TValue *key, StkId val,
                      const TValue *slot) {
  int loop;
  const TValue *tm;
  for (loop = 0; loop < MAXTAGLOOP; loop++) {
    if (slot == NULL) {  /* 't' is not a table? */
      tm = luaT_gettmbyobj(L, t, 0);
      if (__builtin_expect(ttisnil(tm) != 0, 0))  /* no metamethod */
        luaG_typeerror(L, t, "index");
    }
    else {  /* 't' is a table: fasttm(L, mt, TM_INDEX) */
      Table *et = hvalue(t)->metatable;
      tm = (et == NULL) ? NULL
         : (et->flags & (1u << 0)) ? NULL
         : luaT_gettm(et, 0, ((TString **)((char *)L->l_G + 0x118))[0]);
      if (tm == NULL) {  /* no metamethod? */
        setnilvalue(val);
        return;
      }
    }
    if (ttisfunction(tm)) {
      luaT_callTMres(L, tm, t, key, val);
      return;
    }
    t = tm;
    if (ttistable(t)) {
      slot = luaH_get(hvalue(t), key);
      if (!isempty(slot)) {
        setobj(val, slot);
        return;
      }
    }
    else
      slot = NULL;
  }
  luaG_runerror(L, "'__index' chain too long; possible loop");
}
