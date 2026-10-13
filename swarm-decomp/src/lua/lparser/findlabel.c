#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "Labeldesc.h"

static Labeldesc *findlabel (LexState *ls, TString *name) {
  int i;
  Dyndata *dyd = ls->dyd;
  for (i = ls->fs->firstlabel; i < dyd->label.n; i++) {
    Labeldesc *lb = &dyd->label.arr[i];
    if (lb->name == name)
      return lb;
  }
  return NULL;
}
