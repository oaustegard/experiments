#include "lstate.h"
#include "UpVal.h"
#include "fn_luaF_findupval.h"


UpVal *luaF_findupval (lua_State *L, StkId level) {
  UpVal **pp = (UpVal **)&L->openupval;
  UpVal *p;
  while ((p = *pp) != NULL && p->v.p >= level) {
    if (p->v.p == level)
      return p;
    pp = &p->u.open.next;
  }
  return newupval(L, level, pp);
}
