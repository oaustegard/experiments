#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "Labeldesc.h"
#include "fn_luaK_patchlist.h"

static void solvegoto (LexState *ls, int g, Labeldesc *label) {
  int i;
  Labellist *gl = &ls->dyd->gt;
  Labeldesc *gt = &gl->arr[g];
  if (__builtin_expect(gt->nactvar < label->nactvar, 0))
    jumpscopeerror(ls, gt);
  luaK_patchlist(ls->fs, gt->pc, label->pc);
  for (i = g; i < gl->n - 1; i++)
    gl->arr[i] = gl->arr[i + 1];
  gl->n--;
}
