#include <stdlib.h>
#include "lctype.h"
#include "l_str2dloc.h"

static const char *l_str2dloc (const char *s, lua_Number *result, int mode) {
  char *endptr;
  *result = (mode == 'x') ? strtod(s, &endptr) : strtod(s, &endptr);
  if (endptr == s) return NULL;
  while (luai_ctype_[(unsigned char)*endptr + 1] & 8) endptr++;
  return (*endptr == '\0') ? endptr : NULL;
}
