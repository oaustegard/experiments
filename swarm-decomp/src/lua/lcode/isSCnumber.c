#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaV_flttointeger.h"

static int isSCnumber (expdesc *e, int *pi, int *isfloat) {
  lua_Integer i;
  if (e->k == VKINT)
    i = e->u.ival;
  else if (e->k == VKFLT && luaV_flttointeger(e->u.nval, &i, 0))
    *isfloat = 1;
  else
    return 0;
  if (!(e->t != e->f) && (lua_Unsigned)i + 0x7f <= 0xff) {
    *pi = (int)i + 0x7f;
    return 1;
  }
  else
    return 0;
}
