#pragma once
#include "lobject.h"

extern const lu_byte luai_ctype_[];
#define lisprint(c) (luai_ctype_[(c) + 1] & 4)
#define lisxdigit(c) (luai_ctype_[(c) + 1] & 0x10)
