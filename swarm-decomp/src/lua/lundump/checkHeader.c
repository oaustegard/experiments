#include "lundump.h"

#define LUAC_VERSION	0x54
#define LUAC_FORMAT	0
#define LUAC_DATA	"\x19\x93\r\n\x1a\n"
#define LUAC_INT	0x5678
#define LUAC_NUM	370.5

static void checkHeader (LoadState *S) {
  /* skip 1st char (already read and checked) */
  checkliteral(S, "Lua", "not a binary chunk");
  if (loadByte(S) != LUAC_VERSION)
    error(S, "version mismatch");
  if (loadByte(S) != LUAC_FORMAT)
    error(S, "format mismatch");
  checkliteral(S, LUAC_DATA, "corrupted chunk");
  fchecksize(S, 4, "Instruction");
  fchecksize(S, 8, "lua_Integer");
  fchecksize(S, 8, "lua_Number");
  if (loadInteger(S) != LUAC_INT)
    error(S, "integer format mismatch");
  if (loadNumber(S) != LUAC_NUM)
    error(S, "float format mismatch");
}
