#include "FuncState.h"
#include "CREATE_ABCk.h"
#include "lcode_limits.h"
#include "fn_luaX_syntaxerror.h"
#include "luaP_opmodes.h"
#include "patchlistaux.h"
#include "getjump.h"

static void patchlistaux (FuncState *fs, int list, int vtarget, int reg, int dtarget) {
  while (list != -1) {
    int next = getjump(fs, list);
    if (patchtestreg(fs, list, reg))
      fixjump(fs, list, vtarget);
    else
      fixjump(fs, list, dtarget);
    list = next;
  }
}
