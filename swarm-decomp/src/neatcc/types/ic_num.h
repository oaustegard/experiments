#pragma once
#include "ic_ent.h"
/* evaluate table entry idx of tab into *out; nonzero on failure */
int ic_num(struct ic_ent *tab, long idx, long *out);
