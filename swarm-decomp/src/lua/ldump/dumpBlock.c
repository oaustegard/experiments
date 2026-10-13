#include "dumpSize.h"
static void dumpBlock (DumpState *D, const void *b, size_t size) {
  if (D->status == 0 && size > 0)
    D->status = (*D->writer)(D->L, b, size, D->data);
}
