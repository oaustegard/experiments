#include "llex_macros.h"
#include "save.h"

static size_t skip_sep(LexState *ls) {
  size_t count = 0;
  int s = ls->current;
  save(ls, s);
  next(ls);
  while (ls->current == '=') {
    save(ls, '=');
    next(ls);
    count++;
  }
  return (ls->current == s) ? count + 2 : (count == 0) ? 1 : 0;
}
