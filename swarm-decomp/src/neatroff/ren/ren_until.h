#pragma once
#include "wb.h"

static int ren_until(struct wb *w, int (*next)(void), void (*back)(int), char *s, char *t);
