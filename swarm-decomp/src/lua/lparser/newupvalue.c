#include "FuncState.h"
#include "Dyndata.h"
#include "Upvaldesc.h"
#include "Vardesc.h"
#include "luaC_objbarrier.h"
#include "allocupvalue.h"

static int newupvalue (FuncState *fs, TString *name, expdesc *v) {
  Upvaldesc *up = allocupvalue(fs);
  FuncState *prev = fs->prev;
  if (v->k == VLOCAL) {
    up->instack = 1;
    up->idx = v->u.var.ridx;
    up->kind = prev->ls->dyd->actvar.arr[prev->firstlocal + v->u.var.vidx].vd.kind;
  }
  else {
    up->instack = 0;
    up->idx = (lu_byte)v->u.info;
    up->kind = prev->f->upvalues[v->u.info].kind;
  }
  up->name = name;
  luaC_objbarrier(fs->ls->L, fs->f, name);
  return fs->nups - 1;
}
