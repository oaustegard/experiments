#include "lstate.h"
#include "UpVal.h"

void luaF_unlinkupval (UpVal *uv) {
  *uv->u.open.previous = uv->u.open.next;
  if (uv->u.open.next)
    uv->u.open.next->u.open.previous = uv->u.open.previous;
}
