#include "FuncState.h"
#include "expdesc.h"
#include "expkind.h"
#include "TValue.h"
#include "TString.h"
#include "LexState.h"
#include "Dyndata.h"
#include "fn_luaK_exp2const.h"

int luaK_exp2const (FuncState *fs, const expdesc *e, TValue *v) {
  if (e->t != e->f)
    return 0;
  switch (e->k) {
    case VFALSE:
      v->tt_ = 1;
      return 1;
    case VTRUE:
      v->tt_ = 0x11;
      return 1;
    case VNIL:
      v->tt_ = 0;
      return 1;
    case VKSTR: {
      TString *ts = (TString *)e->u.strval;
      v->value_.gc = (GCObject *)ts;
      v->tt_ = ts->tt | 0x40;
      return 1;
    }
    case VCONST: {
      TValue *k = &fs->ls->dyd->actvar.arr[e->u.info].k;
      v->value_ = k->value_;
      v->tt_ = k->tt_;
      return 1;
    }
    default:
      if (e->k == VKFLT) {
        if (v) {
          v->value_.n = e->u.nval;
          v->tt_ = 0x13;
        }
        return 1;
      }
      if (e->k == VKINT) {
        if (v) {
          v->value_.i = e->u.ival;
          v->tt_ = 3;
        }
        return 1;
      }
      return 0;
  }
}
