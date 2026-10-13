#include "dumpSize.h"
#include "Proto.h"
static void dumpCode (DumpState *D, const Proto *f) {
  dumpInt(D, f->sizecode);
  dumpBlock(D, f->code, f->sizecode * sizeof(Instruction));
}
