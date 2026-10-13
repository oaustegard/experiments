#include "llex_macros.h"
#include "save.h"
#include "esccheck.h"
#include "readdecesc.h"

static int readdecesc(LexState *ls) {
  int i;
  int r = 0;
  for (i = 0; i < 3 && lisdigit(ls->current); i++) {
    r = 10 * r + ls->current - '0';
    save_and_next(ls);
  }
  esccheck(ls, r <= 255, "decimal escape too large");
  ls->buff->n -= i;
  return r;
}
