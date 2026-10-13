#include "dumpSize.h"
static void dumpNumber (DumpState *D, lua_Number x) {
  dumpBlock(D, &x, sizeof(x));
}
