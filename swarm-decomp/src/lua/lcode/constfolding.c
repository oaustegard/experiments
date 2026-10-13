#include "FuncState.h"
#include "expdesc.h"
#include "expkind.h"
#include "TValue.h"
#include "LexState.h"
#include "fn_luaO_rawarith.h"
#include "validop.h"
#include "tonumv.h"
#include "constfolding.h"

static int constfolding (FuncState *fs, int op, expdesc *e1, const expdesc *e2) {
  TValue v1, v2, res;
  if (!tonumv(e1, &v1) || !tonumv(e2, &v2) || !validop(op, &v1, &v2))
    return 0;
  luaO_rawarith(fs->ls->L, op, &v1, &v2, &res);
  if (res.tt_ == 3) {
    e1->k = VKINT;
    e1->u.ival = res.value_.i;
  }
  else {
    lua_Number n = res.value_.n;
    if (n != n || n == 0)
      return 0;
    e1->k = VKFLT;
    e1->u.nval = n;
  }
  return 1;
}
