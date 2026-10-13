#include "codebinNoK.h"
#include "swapexps.h"
#include "codebinexpval.h"

static void codebinNoK (FuncState *fs, int opr, expdesc *e1, expdesc *e2, int flip, int line) {
  if (flip)
    swapexps(e1, e2);
  codebinexpval(fs, opr, e1, e2, line);
}
