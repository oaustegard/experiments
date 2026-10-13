#include "lauxlib.h"
#include "byteoffset.h"
#include "codepoint.h"
#include "utfchar.h"
#include "utflen.h"
#include "iter_codes.h"

static const luaL_Reg funcs[] = {
  {"offset", byteoffset},
  {"codepoint", codepoint},
  {"char", utfchar},
  {"len", utflen},
  {"codes", iter_codes},
  /* placeholders */
  {"charpattern", NULL},
  {NULL, NULL}
};
