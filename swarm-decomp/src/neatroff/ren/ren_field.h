#pragma once
#include "wb.h"

static void ren_field(struct wb *w, int (*next)(void), void (*back)(int));
