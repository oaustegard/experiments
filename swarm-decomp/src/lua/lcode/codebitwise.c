#include "codebitwise.h"
#include "swapexps.h"
#include "codebinK.h"
#include "codebinNoK.h"
#include "luaK_exp2K.h"

static void codebitwise (FuncState *fs, int opr, expdesc *e1, expdesc *e2, int line) {
  int flip = 0;
  if (e1->k == VKINT) {
    swapexps(e1, e2);
    flip = 1;
  }
  if (e2->k == VKINT && luaK_exp2K(fs, e2))
    codebinK(fs, opr, e1, e2, flip, line);
  else
    codebinNoK(fs, opr, e1, e2, flip, line);
}
