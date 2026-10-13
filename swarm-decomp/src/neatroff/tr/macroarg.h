#pragma once
#include "sbuf.h"

static int macroarg(struct sbuf *b, int term, int (*rd)(void), void (*put)(int));
