#include "llex_macros.h"
#include "save.h"

static int check_next2(LexState *ls, const char *set) {
  if (ls->current == set[0] || ls->current == set[1]) {
    save(ls, ls->current);
    next(ls);
    return 1;
  } else
    return 0;
}
