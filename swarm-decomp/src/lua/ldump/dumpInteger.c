#include "dumpSize.h"
static void dumpInteger (DumpState *D, lua_Integer x) {
  dumpBlock(D, &x, sizeof(x));
}
