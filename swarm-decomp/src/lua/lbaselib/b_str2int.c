#include <ctype.h>
#include <string.h>
#include "lua_api.h"

static const char *b_str2int (const char *s, int base, lua_Integer *pn) {
  lua_Unsigned n = 0;
  int neg = 0;
  s += strspn(s, " \f\n\r\t\v");
  if (*s == '-') { s++; neg = 1; }
  else if (*s == '+') s++;
  if (!isalnum((unsigned char)*s))
    return NULL;
  do {
    int digit = (isdigit((unsigned char)*s)) ? *s - '0'
                   : (toupper((unsigned char)*s) - 'A') + 10;
    if (digit >= base) return NULL;
    n = n * base + digit;
    s++;
  } while (isalnum((unsigned char)*s));
  s += strspn(s, " \f\n\r\t\v");
  *pn = (lua_Integer)((neg) ? (0u - n) : n);
  return s;
}
