#include "LexState.h"
#include "Labellist.h"
#include "fn_luaM_growaux_.h"

static int newlabelentry (LexState *ls, Labellist *l, TString *name, int line, int pc) {
  int n = l->n;
  l->arr = luaM_growaux_(ls->L, l->arr, n, &l->size, sizeof(Labeldesc), 0x7fff, "labels/gotos");
  l->arr[n].name = name;
  l->arr[n].line = line;
  l->arr[n].nactvar = ls->fs->nactvar;
  l->arr[n].close = 0;
  l->arr[n].pc = pc;
  l->n = n + 1;
  return n;
}
