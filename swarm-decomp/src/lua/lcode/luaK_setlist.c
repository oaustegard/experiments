#include "FuncState.h"
#include "fn_luaK_code.h"
#include "CREATE_ABCk.h"
#include "lopcodes.h"

void luaK_setlist (FuncState *fs, int base, int nelems, int tostore) {
  if (tostore == -1)
    tostore = 0;
  if (nelems <= 255)
    luaK_code(fs, CREATE_ABCk(78, base, tostore, nelems, 0));
  else {
    int extra = nelems / 256;
    nelems %= 256;
    luaK_code(fs, CREATE_ABCk(78, base, tostore, nelems, 1));
    luaK_code(fs, CREATE_Ax(OP_EXTRAARG, extra));
  }
  fs->freereg = base + 1;
}
