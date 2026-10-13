#pragma once
#include "Zio.h"
#include "fn_luaZ_fill.h"

#define zgetc(z) (((z)->n--) > 0 ? (unsigned char)*(z)->p++ : luaZ_fill(z))
#define next(ls) ((ls)->current = zgetc((ls)->z))
#define lislalpha(c) (luai_ctype_[(c) + 1] & 1)
#define lislalnum(c) (luai_ctype_[(c) + 1] & 3)
#define lisdigit(c) (luai_ctype_[(c) + 1] & 2)
#define lisspace(c) (luai_ctype_[(c) + 1] & 8)
#define currIsNewline(ls) ((ls)->current == '\n' || (ls)->current == '\r')
#define save_and_next(ls) (save(ls, (ls)->current), next(ls))

#define isreserved(s) (((s)->tt == 4) && ((s)->extra > 0))
