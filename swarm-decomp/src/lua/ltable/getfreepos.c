#include "Table.h"

static Node *getfreepos (Table *t) {
  if (t->lastfree != NULL) {
    while (t->lastfree > t->node) {
      t->lastfree--;
      if (t->lastfree->u.key_tt == 0)
        return t->lastfree;
    }
  }
  return NULL;
}
