#include "llex_macros.h"
#include "check_next1.h"

static int check_next1(LexState *ls, int c) {
  if (ls->current == c) {
    next(ls);
    return 1;
  } else
    return 0;
}
