#pragma once
#include "lobject.h"
#include "lstate.h"

int luaV_tointeger (const TValue *obj, lua_Integer *p, int mode);
