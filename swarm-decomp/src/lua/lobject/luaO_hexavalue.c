#include "lctype.h"

int luaO_hexavalue (int c) {
  if (luai_ctype_[c + 1] & 2) return c - '0';
  else return ((c | 0x20) - 'a') + 10;
}
