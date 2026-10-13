#include <ctype.h>
#include <locale.h>
#include "lauxlib.h"
#include "read_number.h"

static int read_number (lua_State *L, FILE *f) {
  RN rn;
  int count = 0;
  int hex = 0;
  char decp[2];
  rn.f = f; rn.n = 0;
  decp[0] = localeconv()->decimal_point[0];
  decp[1] = '.';
  flockfile(rn.f);
  do { rn.c = getc_unlocked(rn.f); } while (isspace(rn.c));
  test2(&rn, "-+");
  if (test2(&rn, "00")) {
    if (test2(&rn, "xX")) hex = 1;
    else count = 1;
  }
  count += readdigits(&rn, hex);
  if (test2(&rn, decp))
    count += readdigits(&rn, hex);
  if (count > 0 && test2(&rn, (hex ? "pP" : "eE"))) {
    test2(&rn, "-+");
    readdigits(&rn, 0);
  }
  ungetc(rn.c, rn.f);
  funlockfile(rn.f);
  rn.buff[rn.n] = '\0';
  if (luai_likely(lua_stringtonumber(L, rn.buff)))
    return 1;
  else {
    lua_pushnil(L);
    return 0;
  }
}
