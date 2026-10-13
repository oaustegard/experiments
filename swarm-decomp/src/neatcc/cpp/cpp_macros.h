#pragma once
#include "macro.h"
/* macro table of the cpp TU (static in the target) */
static int mhead[256];
static int mnext[4096];
static int mcount = 1;
static struct macro macros[4096];
