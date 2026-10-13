#include "expdesc.h"
#include "TValue.h"
#include "tonumv.h"

static int tonumv (const expdesc *e, TValue *v) {
  if (e->t != e->f)
    return 0;
  switch (e->k) {
    case VKFLT: v->value_.n = e->u.nval; v->tt_ = 0x13; return 1;
    case VKINT: v->value_.i = e->u.ival; v->tt_ = 3; return 1;
    default: return 0;
  }
}
