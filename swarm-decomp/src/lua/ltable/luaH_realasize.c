#include "Table.h"

#define LTB_BITRAS 0x80

unsigned int luaH_realasize (const Table *t) {
  if (!(t->flags & LTB_BITRAS) || (t->alimit & (t->alimit - 1)) == 0)
    return t->alimit;
  else {
    unsigned int size = t->alimit;
    size |= (size >> 1);
    size |= (size >> 2);
    size |= (size >> 4);
    size |= (size >> 8);
    size |= (size >> 16);
    size++;
    return size;
  }
}
