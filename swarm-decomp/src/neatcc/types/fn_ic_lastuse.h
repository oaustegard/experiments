#pragma once
#include "ic_ent.h"
/* last use index of each int-code temporary; returns the table of last uses */
long *ic_lastuse(struct ic_ent *tab, long n);
