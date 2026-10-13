#include "TString.h"

unsigned int luaS_hashlongstr (TString *ts) {
  if (ts->extra == 0) {
    size_t len = ts->u.lnglen;
    unsigned int h = ts->hash ^ (unsigned int)len;
    size_t l;
    for (l = len; l > 0; l--)
      h ^= ((h << 5) + (h >> 2) + (unsigned char)ts->contents[l - 1]);
    ts->hash = h;
    ts->extra = 1;
  }
  return ts->hash;
}
