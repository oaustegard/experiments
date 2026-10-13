#include "Table.h"
#include "luaH_realasize.h"

static unsigned int setlimittosize (Table *t) {
  t->alimit = luaH_realasize(t);
  t->flags &= ~0x80;
  return t->alimit;
}
