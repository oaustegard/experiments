#include "FuncState.h"
#include "Upvaldesc.h"

static int searchupvalue (FuncState *fs, TString *name) {
  int i;
  Upvaldesc *up = fs->f->upvalues;
  for (i = 0; i < fs->nups; i++) {
    if (up[i].name == name)
      return i;
  }
  return -1;
}
