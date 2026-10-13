#include "expdesc.h"
#include "TValue.h"
#include "lobject_tags.h"
#include "const2exp.h"

static void const2exp(TValue *v, expdesc *e) {
  switch (ttypetag(v)) {
    case LUA_VNUMINT:
      e->k = VKINT;
      e->u.ival = ivalue(v);
      break;
    case LUA_VNUMFLT:
      e->k = VKFLT;
      e->u.nval = v->value_.n;
      break;
    case LUA_VFALSE:
      e->k = VFALSE;
      break;
    case 0x11:
      e->k = VTRUE;
      break;
    case LUA_VNIL:
      e->k = VNIL;
      break;
    case 4:
    case 0x14:
      e->k = VKSTR;
      e->u.strval = v->value_.p;
      break;
    default:
      break;
  }
}
