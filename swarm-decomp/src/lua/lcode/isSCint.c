#include "expdesc.h"

static int isSCint (expdesc *e) {
  return (e->k == VKINT && e->t == e->f && (lua_Unsigned)e->u.ival + 0x7f <= 0xff);
}
