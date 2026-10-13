#include "FuncState.h"
#include "fn_luaX_syntaxerror.h"

void luaK_checkstack (FuncState *fs, int n) {
  int newstack = fs->freereg + n;
  if (newstack > fs->f->maxstacksize) {
    if (newstack >= 255)
      luaX_syntaxerror(fs->ls, "function or expression needs too many registers");
    fs->f->maxstacksize = newstack;
  }
}
