#include "llex_macros.h"
#include "save.h"
#include "readutf8esc.h"
#include "fn_luaO_utf8esc.h"
#include "utf8esc.h"

static void utf8esc(LexState *ls) {
  char buff[8];
  int n = luaO_utf8esc(buff, readutf8esc(ls));
  for (; n > 0; n--)
    save(ls, buff[8 - n]);
}
