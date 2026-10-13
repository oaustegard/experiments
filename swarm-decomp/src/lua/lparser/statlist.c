#include "LexState.h"
#include "statement.h"

static void statlist (LexState *ls) {
  while (!block_follow(ls, 1)) {
    if (ls->t.token == 0x111) {
      statement(ls);
      return;
    }
    statement(ls);
  }
}
