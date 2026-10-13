#pragma once
#include "mem.h"
/* take the buffer of m, leaving m empty; the caller owns the result */
char *mem_get(struct mem *m);
