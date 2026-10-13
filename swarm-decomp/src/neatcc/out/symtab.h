#pragma once
#include "sym.h"
/* the symbol and string tables of this TU (tentative definitions; the
   variable files are not separate because put_sym needs them before
   the variable files are included) */
static struct sym *syms;
static long syms_n;
static long syms_sz;
static char *symstr;
static long symstr_n;
static long symstr_sz;
