#include "lstate.h"

static void checkpointer(GCObject **p, GCObject *o) {
  if (o == *p)
    *p = o->next;
}
