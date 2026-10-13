#include "lctype.h"
#include "fn_luaO_hexavalue.h"

#define MAXBY10 ((lua_Unsigned)0x0ccccccccccccccc)
#define MAXLASTD 7

static const char *l_str2int (const char *s, lua_Integer *result) {
  lua_Unsigned a = 0;
  int empty = 1;
  int neg;
  while (luai_ctype_[(unsigned char)*s + 1] & 8) s++;
  neg = 0;
  if (*s == '-') { s++; neg = 1; }
  else if (*s == '+') s++;
  if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    s += 2;
    for (; luai_ctype_[(unsigned char)*s + 1] & 0x10; s++) {
      a = a * 16 + luaO_hexavalue(*s);
      empty = 0;
    }
  }
  else {
    for (; luai_ctype_[(unsigned char)*s + 1] & 2; s++) {
      int d = *s - '0';
      if (a >= MAXBY10 && (a > MAXBY10 || d > MAXLASTD + neg))
        return NULL;
      a = a * 10 + d;
      empty = 0;
    }
  }
  while (luai_ctype_[(unsigned char)*s + 1] & 8) s++;
  if (empty || *s != '\0') return NULL;
  else {
    *result = (lua_Integer)(neg ? 0u - a : a);
    return s;
  }
}
