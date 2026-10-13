#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "Dyndata.h"
#include "Labeldesc.h"

static int solvegotos (LexState *ls, Labeldesc *lb) {
  Labellist *gl = &ls->dyd->gt;
  int i = ls->fs->bl->firstgoto;
  int needsclose = 0;
  while (i < gl->n) {
    if (gl->arr[i].name == lb->name) {
      needsclose |= gl->arr[i].close;
      solvegoto(ls, i, lb);
    }
    else
      i++;
  }
  return needsclose;
}
