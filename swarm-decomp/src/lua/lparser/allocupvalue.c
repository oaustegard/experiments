#include "FuncState.h"
#include "Upvaldesc.h"
#include "fn_luaM_growaux_.h"
#include "errorlimit.h"

static Upvaldesc *allocupvalue (FuncState *fs) {
  Proto *f = fs->f;
  int oldsize = f->sizeupvalues;
  if (fs->nups + 1 > 255)
    errorlimit(fs, 255, "upvalues");
  f->upvalues = luaM_growaux_(fs->ls->L, f->upvalues, fs->nups, &f->sizeupvalues,
                              sizeof(Upvaldesc), 255, "upvalues");
  while (oldsize < f->sizeupvalues)
    ((Upvaldesc *)f->upvalues)[oldsize++].name = NULL;
  return &((Upvaldesc *)f->upvalues)[fs->nups++];
}
