#include "linkgclist_.h"
#include "lgc_defs.h"

static void linkgclist_(GCObject *o, GCObject **pnext, GCObject **list) {
  *pnext = *list;
  *list = o;
  set2gray(o);
}
