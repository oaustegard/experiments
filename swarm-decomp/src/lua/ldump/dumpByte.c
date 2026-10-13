#include "dumpSize.h"
static void dumpByte (DumpState *D, int y) {
  lu_byte x = (lu_byte)y;
  dumpBlock(D, &x, 1);
}
