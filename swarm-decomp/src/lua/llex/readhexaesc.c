#include "llex_macros.h"
#include "gethexa.h"
#include "readhexaesc.h"

static int readhexaesc(LexState *ls) {
  int r = gethexa(ls);
  r = (r << 4) + gethexa(ls);
  ls->buff->n -= 2;
  return r;
}
