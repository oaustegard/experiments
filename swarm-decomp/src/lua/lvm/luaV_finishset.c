#include "lstate.h"
#include "ltags.h"
#include "Table.h"
#include "TString.h"
#include "fn_luaT_gettmbyobj.h"
#include "fn_luaT_callTM.h"
#include "fn_luaH_get.h"
#include "fn_luaH_finishset.h"
#include "fn_luaC_barrierback_.h"
#include "fn_luaG_typeerror.h"
#include "fn_luaG_runerror.h"
#include "TMS.h"

#define MAXTAGLOOP 2000
#define maskflags (~(~0u << (TM_EQ + 1)))

void luaV_finishset (lua_State *L, const TValue *t, TValue *key, TValue *val,
                     const TValue *slot) {
  int loop = MAXTAGLOOP;
  do {
    const TValue *tm;
    if (slot != NULL) {
      Table *h = hvalue(t);
      tm = fasttm(L, h->metatable, TM_NEWINDEX);
      if (tm == NULL) {
        luaH_finishset(L, h, key, slot, val);
        h->flags &= ~maskflags;
        if (iscollectable(val) && isblack(h) && iswhite(gcvalue(val)))
          luaC_barrierback_(L, (GCObject *)h);
        return;
      }
    }
    else {
      tm = luaT_gettmbyobj(L, t, TM_NEWINDEX);
      if (ttisnil(tm))
        luaG_typeerror(L, t, "index");
    }
    if (ttisfunction(tm)) {
      luaT_callTM(L, tm, t, key, val);
      return;
    }
    t = tm;
    if (ttistable(t)) {
      slot = luaH_get(hvalue(t), key);
      if (!isempty(slot)) {
        setobj((TValue *)slot, val);
        if (iscollectable(val) && isblack(hvalue(t)) && iswhite(gcvalue(val)))
          luaC_barrierback_(L, gcvalue(t));
        return;
      }
    }
    else
      slot = NULL;
  } while (--loop != 0);
  luaG_runerror(L, "'__newindex' chain too long; possible loop");
}
