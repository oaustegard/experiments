#pragma once
#include "ic_ent.h"
/* evaluate a symbol entry idx of tab into *a and *b; nonzero on failure */
int ic_sym(struct ic_ent *tab, long idx, long *a, long *b);
