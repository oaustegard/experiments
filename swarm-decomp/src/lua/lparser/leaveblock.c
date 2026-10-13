#include "FuncState.h"
#include "BlockCnt.h"
#include "Dyndata.h"
#include "reglevel.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaK_codeABCk.h"
#include "createlabel.h"

static void leaveblock (FuncState *fs) {
  BlockCnt *bl = fs->bl;
  LexState *ls = fs->ls;
  int hasclose = 0;
  int stklevel = reglevel(fs, bl->nactvar);
  removevars(fs, bl->nactvar);
  if (bl->isloop)
    hasclose = createlabel(ls, luaS_newlstr(ls->L, "break", 5), 0, 0);
  if (!hasclose && bl->previous && bl->upval)
    luaK_codeABCk(fs, 0x36, stklevel, 0, 0, 0);
  fs->freereg = stklevel;
  ls->dyd->label.n = bl->firstlabel;
  fs->bl = bl->previous;
  if (bl->previous)
    movegotosout(fs, bl);
  else {
    if (bl->firstgoto < ls->dyd->gt.n)
      undefgoto(ls, &ls->dyd->gt.arr[bl->firstgoto]);
  }
}
