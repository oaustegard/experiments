#include "FuncState.h"
#include "expdesc.h"
#include "addk.h"
#include "luaK_numberK.h"
#include "luaK_exp2K.h"

static int luaK_exp2K (FuncState *fs, expdesc *e) {
  if (e->t == e->f) {
    int info;
    switch (e->k) {
      case VTRUE: {
        TValue v;
        v.tt_ = 0x11;
        info = addk(fs, &v, &v);
        break;
      }
      case VFALSE: {
        TValue v;
        v.tt_ = 1;
        info = addk(fs, &v, &v);
        break;
      }
      case VNIL: {
        TValue k, v;
        v.tt_ = 0;
        k.tt_ = 0x45;
        k.value_.gc = (GCObject *)fs->ls->h;
        info = addk(fs, &k, &v);
        break;
      }
      case VKINT: {
        TValue v;
        v.value_.i = e->u.ival;
        v.tt_ = 3;
        info = addk(fs, &v, &v);
        break;
      }
      case VKFLT: {
        info = luaK_numberK(fs, e->u.nval);
        break;
      }
      case VKSTR: {
        TValue o;
        o.value_.gc = (GCObject *)e->u.strval;
        o.tt_ = ((GCObject *)e->u.strval)->tt | 0x40;
        info = addk(fs, &o, &o);
        break;
      }
      case VK: {
        info = e->u.info;
        break;
      }
      default:
        return 0;
    }
    if (info <= 0xff) {
      e->k = VK;
      e->u.info = info;
      return 1;
    }
  }
  return 0;
}
