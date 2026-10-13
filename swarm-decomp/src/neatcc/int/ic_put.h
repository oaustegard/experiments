#pragma once
#include "ic_ent.h"
/* append an entry (op, a, b, c) to ic; returns the new entry */
static struct ic_ent *ic_put(long op, long a, long b, long c);
