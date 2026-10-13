#include "lstate.h"
#include "Proto.h"
#include "LocVar.h"
#include "TString.h"

const char *luaF_getlocalname (const Proto *f, int local_number, int pc) {
  int i;
  for (i = 0; i < f->sizelocvars && f->locvars[i].startpc <= pc; i++) {
    if (pc < f->locvars[i].endpc) {
      local_number--;
      if (local_number == 0)
        return f->locvars[i].varname->contents;
    }
  }
  return NULL;
}
