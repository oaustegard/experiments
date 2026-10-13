#include "FuncState.h"
#include "Dyndata.h"
#include "Vardesc.h"

static int searchvar (FuncState *fs, TString *n, expdesc *var) {
  int i;
  for (i = (int)fs->nactvar - 1; i >= 0; i--) {
    Vardesc *vd = &fs->ls->dyd->actvar.arr[fs->firstlocal + i];
    if (n == vd->vd.name) {
      if (vd->vd.kind == 3) {
        var->f = var->t = -1;
        var->k = VCONST;
        var->u.info = fs->firstlocal + i;
      }
      else {
        var->f = var->t = -1;
        var->k = VLOCAL;
        var->u.var.vidx = (unsigned short)i;
        var->u.var.ridx = vd->vd.ridx;
      }
      return var->k;
    }
  }
  return -1;
}
