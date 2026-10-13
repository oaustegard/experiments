#include "FuncState.h"
#include "BlockCnt.h"
#include "Dyndata.h"
#include "reglevel.h"

static void movegotosout (FuncState *fs, BlockCnt *bl) {
  int i;
  Labellist *gl = &fs->ls->dyd->gt;
  for (i = bl->firstgoto; i < gl->n; i++) {
    Labeldesc *gt = &gl->arr[i];
    if (reglevel(fs, gt->nactvar) > reglevel(fs, bl->nactvar))
      gt->close |= bl->upval;
    gt->nactvar = bl->nactvar;
  }
}
