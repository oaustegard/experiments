#include "mainpositionTV.h"

static Node *mainpositionfromnode (const Table *t, Node *nd) {
  TValue key;
  key.value_ = nd->u.key_val;
  key.tt_ = nd->u.key_tt;
  return mainpositionTV(t, &key);
}
