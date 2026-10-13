#include "lobject.h"
#include "Table.h"

int countint (lua_Integer key, unsigned int *nums);

static int numusehash (const Table *t, unsigned int *nums, unsigned int *pna) {
  int totaluse = 0;
  int ause = 0;
  int i = 1 << t->lsizenode;
  while (i--) {
    Node *n = &t->node[i];
    if ((n->i_val.tt_ & 0xf) != 0) {
      if (n->u.key_tt == 0x03)
        ause += countint(n->u.key_val.i, nums);
      totaluse++;
    }
  }
  *pna += ause;
  return totaluse;
}
