#include "dumpSize.h"
static void dumpHeader (DumpState *D) {
  dumpBlock(D, "\x1bLua", 4);
  dumpByte(D, 0x54);
  dumpByte(D, 0);
  dumpBlock(D, "\x19\x93\r\n\x1a\n", 6);
  dumpByte(D, sizeof(Instruction));
  dumpByte(D, sizeof(lua_Integer));
  dumpByte(D, sizeof(lua_Number));
  dumpInteger(D, 0x5678);
  dumpNumber(D, 370.5);
}
