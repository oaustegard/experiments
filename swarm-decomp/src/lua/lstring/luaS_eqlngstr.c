#include <string.h>
#include "TString.h"

int luaS_eqlngstr (TString *a, TString *b) {
  size_t len = a->u.lnglen;
  return (a == b) || ((len == b->u.lnglen) && (memcmp(a->contents, b->contents, len) == 0));
}
