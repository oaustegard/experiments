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

/* fasttm(L, mt, TM_INDEX) with tmname[0] at l_G+0x118 */
static const TValue *fasttm_index (lua_State *L, Table *mt) {
  if (mt == NULL || (mt->flags & (1u << 0)))
    return NULL;
  return luaT_gettm(mt, 0, ((TString **)((char *)L->l_G + 0x118))[0]);
}

void luaV_finishget (lua_State *L, const TValue *t, TValue *key, StkId val,
                      const TValue *slot) {
  int loop;
  const TValue *tm;
  for (loop = 0; loop < MAXTAGLOOP; loop++) {
    if (slot == NULL) {
      tm = luaT_gettmbyobj(L, t, 0);
      if (ttisnil(tm))
        luaG_typeerror(L, t, "index");
    }
    else {
      tm = fasttm_index(L, hvalue(t)->metatable);
      if (tm == NULL) {
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
