#include "dumpSize.h"
#include "Proto.h"
static void dumpFunction(DumpState *D, const Proto *f, TString *psource);
static void dumpProtos (DumpState *D, const Proto *f) {
  int i, n = f->sizep;
  dumpInt(D, n);
  for (i = 0; i < n; i++)
    dumpFunction(D, f->p[i], f->source);
}
