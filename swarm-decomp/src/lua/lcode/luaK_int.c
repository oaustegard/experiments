#include "FuncState.h"
#include "fn_luaK_code.h"
#include "lopcodes.h"
#include "addk.h"

void luaK_int (FuncState *fs, int reg, lua_Integer i) {
  if ((lua_Unsigned)i + 0xffff <= MAXARG_Bx)
    luaK_code(fs, CREATE_ABx(1, reg, (int)i + 0xffff));
  else
    luaK_codek(fs, reg, luaK_intK(fs, i));
}
