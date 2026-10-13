#include "FuncState.h"
#include "fn_luaK_code.h"
#include "fn_luaV_flttointeger.h"
#include "luaK_numberK.h"
#include "luaK_codek.h"
#include "lopcodes.h"

static void luaK_float (FuncState *fs, int reg, lua_Number f) {
  lua_Integer fi;
  if (luaV_flttointeger(f, &fi, 0) &&
      (lua_Unsigned)fi + 0xffff <= 0x1ffff)
    luaK_code(fs, CREATE_ABx(2, reg, (int)fi + 0xffff));
  else
    luaK_codek(fs, reg, luaK_numberK(fs, f));
}
