#include "lstate.h"
#include "cclosure.h"
#include "Udata.h"
#include "TString.h"
#include "UpVal.h"
#include "LClosure.h"
#include "fn_luaS_remove.h"
#include "fn_luaM_free_.h"
#include "fn_luaH_free.h"
#include "fn_luaE_freethread.h"
#include "fn_luaF_unlinkupval.h"
#include "fn_luaF_freeproto.h"

static void freeobj (lua_State *L, GCObject *o) {
  switch (o->tt) {
    case 4: {
      TString *ts = (TString *)o;
      luaS_remove(L, ts);
      luaM_free_(L, ts, (size_t)0x19 + ts->shrlen);
      break;
    }
    case 5: {
      luaH_free(L, (Table *)o);
      break;
    }
    case 6: {
      LClosure *cl = (LClosure *)o;
      luaM_free_(L, cl, 0x20 + 8 * cl->nupvalues);
      break;
    }
    case 7: {
      Udata *u = (Udata *)o;
      size_t sz = (u->nuvalue == 0 ? 0x20 : 0x28 + 16 * (size_t)u->nuvalue) + u->len;
      luaM_free_(L, o, sz);
      break;
    }
    case 8: {
      luaE_freethread(L, (lua_State *)o);
      break;
    }
    case 9: {
      UpVal *uv = (UpVal *)o;
      if (uv->v.p != &uv->u.value)
        luaF_unlinkupval(uv);
      luaM_free_(L, uv, sizeof(UpVal));
      break;
    }
    case 0xa: {
      luaF_freeproto(L, (Proto *)o);
      break;
    }
    case 0x14: {
      TString *ts = (TString *)o;
      luaM_free_(L, ts, 0x19 + ts->u.lnglen);
      break;
    }
    case 0x26: {
      CClosure *cl = (CClosure *)o;
      luaM_free_(L, cl, 0x20 + 16 * cl->nupvalues);
      break;
    }
    default:
      break;
  }
}
