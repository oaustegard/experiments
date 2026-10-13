#pragma once
#include "sbuf.h"

static void read_until(struct sbuf *b, char *end, int (*next)(), void (*back)());
