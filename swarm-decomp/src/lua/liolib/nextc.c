#include "lauxlib.h"

static int nextc (RN *rn) {
  if (rn->n < L_MAXLENNUM) {
    rn->buff[rn->n++] = rn->c;
    rn->c = getc_unlocked(rn->f);
    return 1;
  }
  else {
    rn->buff[0] = '\0';
    return 0;
  }
}
