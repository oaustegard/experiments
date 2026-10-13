#include "dumpString.h"
static void dumpString (DumpState *D, const TString *s) {
  if (s == NULL)
    dumpSize(D, 0);
  else {
    size_t size = (s->shrlen != 0xff) ? s->shrlen : s->u.lnglen;
    const char *str = s->contents;
    dumpSize(D, size + 1);
    dumpBlock(D, str, size);
  }
}
