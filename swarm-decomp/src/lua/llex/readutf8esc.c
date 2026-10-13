#include "llex_macros.h"
#include "save.h"
#include "esccheck.h"
#include "gethexa.h"
#include "fn_luaO_hexavalue.h"
#include "readutf8esc.h"

static unsigned long readutf8esc(LexState *ls) {
  unsigned long r;
  int i = 4;
  save_and_next(ls);
  esccheck(ls, ls->current == '{', "missing '{'");
  r = gethexa(ls);
  while (save_and_next(ls), lisxdigit(ls->current)) {
    i++;
    esccheck(ls, r <= (0x7FFFFFFFu >> 4), "UTF-8 value too large");
    r = (r << 4) + luaO_hexavalue(ls->current);
  }
  esccheck(ls, ls->current == '}', "missing '}'");
  next(ls);
  ls->buff->n -= i;
  return r;
}
