#include "fn_luaO_utf8esc.h"

#define UTF8BUFFSZ 8

int luaO_utf8esc (char *buff, unsigned long x) {
  int n = 1;
  if (x < 0x80)
    buff[UTF8BUFFSZ - 1] = (char)x;
  else {
    unsigned int mfb = 0x3f;
    do {
      buff[UTF8BUFFSZ - (n++)] = (char)(0x80 | (x & 0x3f));
      x >>= 6;
      mfb >>= 1;
    } while (x > mfb);
    buff[UTF8BUFFSZ - n] = (char)((~mfb << 1) | x);
  }
  return n;
}
