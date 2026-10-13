#include "lstate.h"
#include "Table.h"

void luaH_resize (lua_State *L, Table *t, unsigned int newasize, unsigned int nhsize) {
  unsigned int i;
  unsigned int oldasize = setlimittosize(t);
  TValue *newarray;
  Node *nlast;
  Node *nnode;
  lu_byte nls;
  if (nhsize == 0) {
    nlast = NULL;
    nnode = (Node *)&dummynode_;
    nls = 0;
  }
  else {
    int lsize = luaO_ceillog2(nhsize);
    int size;
    if (lsize > 30)
      luaG_runerror(L, "table overflow");
    size = 1 << lsize;
    nnode = (Node *)luaM_malloc_(L, size * sizeof(Node), 0);
    for (i = 0; i < (unsigned int)size; i++) {
      Node *n = &nnode[i];
      n->u.next = 0;
      n->u.key_tt = 0;
      n->i_val.tt_ = 0x10;
    }
    nls = (lu_byte)lsize;
    nlast = &nnode[size];
  }
  if (newasize < oldasize) {
    lu_byte a = t->lsizenode;
    Node *b = t->node;
    Node *c = t->lastfree;
    t->alimit = newasize;
    t->lsizenode = nls;
    t->node = nnode;
    t->lastfree = nlast;
    for (i = newasize; i < oldasize; i++) {
      if (t->array[i].tt_ & 0xf) {
        lua_Integer key = i + 1;
        const TValue *p = luaH_getint(t, key);
        if (p->tt_ == 0x20) {
          TValue k;
          k.value_.i = key;
          k.tt_ = 0x03;
          luaH_newkey(L, t, &k, &t->array[i]);
        }
        else {
          TValue *q = (TValue *)p;
          q->value_ = t->array[i].value_;
          q->tt_ = t->array[i].tt_;
        }
      }
    }
    t->alimit = oldasize;
    nls = t->lsizenode;
    nnode = t->node;
    nlast = t->lastfree;
    t->lsizenode = a;
    t->node = b;
    t->lastfree = c;
  }
  newarray = (TValue *)luaM_realloc_(L, t->array, oldasize * sizeof(TValue), newasize * sizeof(TValue));
  if (newarray == NULL && newasize > 0) {
    if (nlast != NULL)
      luaM_free_(L, nnode, (1 << nls) * sizeof(Node));
    luaD_throw(L, 4);
  }
  {
    lu_byte a = t->lsizenode;
    Node *b = t->node;
    Node *c = t->lastfree;
    t->lsizenode = nls;
    t->node = nnode;
    t->lastfree = nlast;
    nls = a;
    nnode = b;
    nlast = c;
  }
  t->array = newarray;
  t->alimit = newasize;
  for (i = oldasize; i < newasize; i++)
    t->array[i].tt_ = 0x10;
  {
    int j;
    int size = 1 << nls;
    for (j = 0; j < size; j++) {
      Node *old = &nnode[j];
      if (old->i_val.tt_ & 0xf) {
        TValue k;
        const TValue *slot;
        k.value_ = old->u.key_val;
        k.tt_ = old->u.key_tt;
        slot = luaH_get(t, &k);
        if (slot->tt_ == 0x20)
          luaH_newkey(L, t, &k, &old->i_val);
        else {
          TValue *s = (TValue *)slot;
          s->value_ = old->i_val.value_;
          s->tt_ = old->i_val.tt_;
        }
      }
    }
  }
  if (nlast != NULL)
    luaM_free_(L, nnode, (1 << nls) * sizeof(Node));
}
