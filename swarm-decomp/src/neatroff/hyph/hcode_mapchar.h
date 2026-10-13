#pragma once
#include "dict.h"

/* data used by hcode_mapchar; sorts before hcodedict.c in the generated TU */
static struct dict *hcodedict;
static char hcodedst[512][32];

static char *hcode_mapchar(char *s);
