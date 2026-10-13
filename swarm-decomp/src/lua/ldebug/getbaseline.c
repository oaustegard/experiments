#include "lstate.h"
#include "Proto.h"
#include "AbsLineInfo.h"

#define MAXIWTHABS	128

static int getbaseline (const Proto *f, int pc, int *basepc) {
  if (f->sizeabslineinfo == 0 || pc < f->abslineinfo[0].pc) {
    /* no absolute info or 'pc' before the first one: start from the beginning */
    *basepc = -1;
    return f->linedefined;
  }
  else {
    int i = (unsigned int)pc / MAXIWTHABS - 1;  /* get an estimate */
    while (i + 1 < f->sizeabslineinfo && pc >= f->abslineinfo[i + 1].pc)
      i++;  /* low estimate; adjust it */
    *basepc = f->abslineinfo[i].pc;
    return f->abslineinfo[i].line;
  }
}
