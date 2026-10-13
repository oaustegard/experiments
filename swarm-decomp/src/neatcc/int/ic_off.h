#pragma once
#include "ic_ent.h"
/* resolve entry idx of tab into *a and *b (offsets); nonzero on failure */
static int ic_off(struct ic_ent *tab, long idx, long *a, long *b);
