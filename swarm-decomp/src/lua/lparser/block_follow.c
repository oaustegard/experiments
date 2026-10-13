#include "LexState.h"

static int block_follow (LexState *ls, int withuntil) {
  switch (ls->t.token) {
    case 0x103: case 0x104: case 0x105: case 0x120:
      return 1;
    case 0x114:
      return withuntil;
    default:
      return 0;
  }
}
