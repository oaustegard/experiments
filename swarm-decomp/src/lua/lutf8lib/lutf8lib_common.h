#pragma once
#include "lua.h"

#define MAXUNICODE	0x10FFFFu
#define MAXUTF		0x7FFFFFFFu
#define MSGInvalid	"invalid UTF-8 code"
#define iscont(c)	(((c) & 0xC0) == 0x80)
#define iscontp(p)	iscont(*(p))

typedef unsigned int utfint;
