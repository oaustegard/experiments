#include "lauxlib.h"
#include "prepbuffsize.h"
#include <string.h>

void luaL_addlstring (luaL_Buffer *B, const char *s, size_t l) {
  if (l > 0) {
    char *b = prepbuffsize(B, l, -1);
    memcpy(b, s, l);
    B->n += l;
  }
}
