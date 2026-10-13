#include "luaK_numberK.h"
#include "addk.h"
#include "fn_luaV_flttointeger.h"
#include <math.h>

static int luaK_numberK (FuncState *fs, lua_Number r) {
  TValue o;
  TValue kv;
  lua_Integer ik;
  o.value_.n = r;
  o.tt_ = 0x13;
  if (!luaV_flttointeger(r, &ik, 0))
    return addk(fs, &o, &o);
  else {
    const lua_Number q = ldexp(1.0, -52);
    const lua_Number k = (ik == 0) ? q : r + r * q;
    kv.value_.n = k;
    kv.tt_ = 0x13;
    return addk(fs, &kv, &o);
  }
}
