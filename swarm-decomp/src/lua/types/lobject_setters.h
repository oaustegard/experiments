#pragma once
#include "lobject.h"

/* value copies of lobject.h (Lua 5.4): copy the value and the tag field by field */
#define setobjs2s(L,o1,o2) \
  { TValue *io1 = (o1); const TValue *io2 = (o2); \
    io1->value_ = io2->value_; io1->tt_ = io2->tt_; }
#define setnilvalue(obj)	((obj)->tt_ = 0)
#define s2v(o)	(o)
