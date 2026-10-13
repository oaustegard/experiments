#include "FuncState.h"
#include "Dyndata.h"
#include "Vardesc.h"

static int reglevel (FuncState *fs, int nvar) {
  while (nvar-- > 0) {
    Vardesc *vd = &fs->ls->dyd->actvar.arr[fs->firstlocal + nvar];
    if (vd->vd.kind != 3)
      return vd->vd.ridx + 1;
  }
  return 0;
}
