#pragma once
#include "lobject.h"

int luaV_tointegerns (const TValue *obj, lua_Integer *p, int mode);
