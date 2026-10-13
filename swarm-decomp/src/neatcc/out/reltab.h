#pragma once
#include "rel.h"
/* the relocation tables of this TU (tentative definitions, as symtab.h) */
static struct rel *dsrel;
static long dsrel_n;
static long dsrel_sz;
static struct rel *csrel;
static long csrel_n;
static long csrel_sz;
