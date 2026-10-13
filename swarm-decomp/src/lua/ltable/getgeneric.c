#include "Table.h"

static Node *mainpositionTV (const Table *t, const TValue *key);
static int equalkey (const TValue *k1, const Node *n2, int deadok);

static const TValue *getgeneric (Table *t, const TValue *key, int deadok) {
  Node *n = mainpositionTV(t, key);
  for (;;) {
    if (equalkey(key, n, deadok))
      return &n->i_val;
    else {
      int nx = n->u.next;
      if (nx == 0)
        return &absentkey;
      n += nx;
    }
  }
}
