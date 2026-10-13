#pragma once
#include "type.h"
#include "decl.h"
/* p: the decl or the loc slot; fn: the element initialiser (globalinit or localinit) */
static void initexpr(struct type *t, int local, void *p, void (*fn)());
