#include "lstate.h"
#include "Table.h"

static void setnodevector (lua_State *L, Table *t, unsigned int size) {
  if (size == 0) {  /* no elements to hash part? */
    t->node = (Node *)&dummynode_;  /* use common 'dummynode' */
    t->lsizenode = 0;
    t->lastfree = NULL;  /* signal that it is using dummy node */
  }
  else {
    int i;
    int lsize = luaO_ceillog2(size);
    if (lsize > 30)
      luaG_runerror(L, "table overflow");
    size = 1 << lsize;
    t->node = (Node *)luaM_malloc_(L, size * sizeof(Node), 0);
    for (i = 0; i < (int)size; i++) {
      Node *n = &t->node[i];
      n->u.next = 0;
      n->u.key_tt = 0;
      n->i_val.tt_ = 0x10;
    }
    t->lsizenode = (lu_byte)lsize;
    t->lastfree = &t->node[size];  /* all positions are free */
  }
}
