#include "FuncState.h"
#include "Dyndata.h"
#include "Vardesc.h"

static Vardesc *getlocalvardesc (FuncState *fs, int vidx) {
  return &fs->ls->dyd->actvar.arr[fs->firstlocal + vidx];
}
