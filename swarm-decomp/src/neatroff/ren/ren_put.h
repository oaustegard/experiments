#pragma once
#include "wb.h"

static void ren_put(struct wb *w, char *s, int (*next)(void), void (*back)(int));
