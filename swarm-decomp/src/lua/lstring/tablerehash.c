#include "TString.h"

static void tablerehash (TString **vect, int osize, int nsize) {
  int i;
  for (i = osize; i < nsize; i++)
    vect[i] = NULL;
  for (i = 0; i < osize; i++) {
    TString *p = vect[i];
    vect[i] = NULL;
    while (p) {
      TString *hnext = p->u.hnext;
      unsigned int h = p->hash & (nsize - 1);
      p->u.hnext = vect[h];
      vect[h] = p;
      p = hnext;
    }
  }
}
