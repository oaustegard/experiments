#include "lstate.h"
#include "LClosure.h"
#include "UpVal.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

static UpVal **getupvalref (lua_State *L, int fidx, int n, LClosure **pf) {
  static const UpVal *const nullup = NULL;
  LClosure *f;
  TValue *fi = index2value(L, fidx);
  f = (LClosure *)fi->value_.gc;
  if (pf) *pf = f;
  if (1 <= n && n <= f->p->sizeupvalues)
    return &f->upvals[n-1];
  else
    return (UpVal **)&nullup;
}
