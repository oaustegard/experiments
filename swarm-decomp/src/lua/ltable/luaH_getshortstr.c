#include "Table.h"
#include "TString.h"

const TValue *luaH_getshortstr (Table *t, TString *key) {
  Node *n = &t->node[(int)(key->hash & ((1 << t->lsizenode) - 1))];
  for (;;) {
    if (n->u.key_tt == 0x44 && (TString *)n->u.key_val.gc == key)
      return &n->i_val;
    else {
      int nx = n->u.next;
      if (nx == 0)
        return &absentkey;
      n += nx;
    }
  }
}
