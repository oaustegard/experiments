#include "llex_macros.h"
#include "save.h"
#include "lexerror.h"
#include "esccheck.h"

static void esccheck(LexState *ls, int c, const char *msg) {
  if (!c) {
    if (ls->current != -1) {
      save(ls, ls->current);
      next(ls);
    }
    lexerror(ls, msg, 0x124);
  }
}
