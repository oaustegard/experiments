#pragma once
#include "wb.h"

static void ren_tab(struct wb *w, char *s, int (*next)(void), void (*back)(int));
