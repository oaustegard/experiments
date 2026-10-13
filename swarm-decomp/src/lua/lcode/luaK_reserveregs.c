#include "FuncState.h"
#include "fn_luaK_checkstack.h"

void luaK_reserveregs (FuncState *fs, int n) {
  luaK_checkstack(fs, n);
  fs->freereg += n;
}
