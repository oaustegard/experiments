#include "lstate.h"
#include "dothecall.h"
#include "fn_luaT_gettmbyobj.h"
#include "fn_luaD_pcall.h"
#include "fn_luaE_warnerror.h"

static void GCTM (lua_State *L) {
  global_State *g = L->l_G;
  const TValue *tm;
  TValue v;
  GCObject *o = g->tobefnz;
  g->tobefnz = o->next;
  o->next = g->allgc;
  g->allgc = o;
  o->marked &= ~0x40;
  if (g->gcstate >= 3 && g->gcstate <= 6)
    o->marked = (lu_byte)((o->marked & ~0x78) | (g->currentwhite & 0x18));
  else if ((o->marked & 7) == 3)
    g->firstold1 = o;
  v.value_.gc = o;
  v.tt_ = (lu_byte)(o->tt | 0x40);
  tm = luaT_gettmbyobj(L, &v, 2);
  if ((tm->tt_ & 0xf) != 0) {
    int status;
    lu_byte oldah = L->allowhook;
    int oldgcstp = g->gcstp;
    g->gcstp |= 2;
    L->allowhook = 0;
    {
      TValue *t1 = L->top++;
      t1->value_ = tm->value_;
      t1->tt_ = tm->tt_;
    }
    {
      TValue *t2 = L->top++;
      t2->value_ = v.value_;
      t2->tt_ = v.tt_;
    }
    L->ci->callstatus |= 0x80;
    status = luaD_pcall(L, dothecall, NULL,
                        (char *)(L->top - 2) - (char *)L->stack, 0);
    L->ci->callstatus &= ~0x80;
    L->allowhook = oldah;
    g->gcstp = oldgcstp;
    if (__builtin_expect(status != 0, 0)) {
      luaE_warnerror(L, "__gc");
      L->top--;
    }
  }
}
