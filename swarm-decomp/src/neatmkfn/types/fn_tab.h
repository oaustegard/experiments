#pragma once
#include "tab.h"
struct tab *tab_alloc(int n);
void tab_put(struct tab *t, const char *key, const void *val);
