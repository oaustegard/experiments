#include "lobject.h"
#include "Table.h"

unsigned int luaH_realasize (const Table *t);
const TValue *luaH_getint (Table *t, lua_Integer key);

lua_Unsigned luaH_getn (Table *t) {
  unsigned int limit = t->alimit;
  if (limit > 0 && (t->array[limit - 1].tt_ & 0xf) == 0) {
    if (limit >= 2 && (t->array[limit - 2].tt_ & 0xf) != 0) {
      if ((!(t->flags & 0x80) || (t->alimit & (t->alimit - 1)) == 0) && ((limit - 1) & (limit - 2)) != 0) {
        t->alimit = limit - 1;
        t->flags |= 0x80;
        return t->alimit;
      }
      else
        return limit - 1;
    }
    else {
      unsigned int border = binsearch(t->array, 0, limit);
      if ((!(t->flags & 0x80) || (t->alimit & (t->alimit - 1)) == 0) && border > luaH_realasize(t) / 2) {
        t->alimit = border;
        t->flags |= 0x80;
      }
      return border;
    }
  }
  if (!((!(t->flags & 0x80)) || (limit & (limit - 1)) == 0)) {
    if ((t->array[limit].tt_ & 0xf) == 0)
      return limit;
    limit = luaH_realasize(t);
    if ((t->array[limit - 1].tt_ & 0xf) == 0) {
      unsigned int border = binsearch(t->array, t->alimit, limit);
      t->alimit = border;
      return border;
    }
  }
  if (t->lastfree == NULL || (luaH_getint(t, limit + 1)->tt_ & 0xf) == 0)
    return limit;
  else
    return hash_search(t, limit);
}
