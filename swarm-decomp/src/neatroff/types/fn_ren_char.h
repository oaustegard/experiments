#pragma once
#include "wb.h"

int ren_char(struct wb *w, int (*next)(void), void (*back)(int));
