#include "lstate.h"
#include "Proto.h"

int luaG_getfuncline (const Proto *f, int pc) {
  if (f->lineinfo == NULL)  /* no debug information? */
    return -1;
  else {
    int basepc;
    int baseline = getbaseline(f, pc, &basepc);
    while (basepc++ < pc) {  /* walk until given instruction */
      baseline += f->lineinfo[basepc];  /* correct line */
    }
    return baseline;
  }
}
