#pragma once
#include "lstate.h"

void luaV_finishset (lua_State *L, const TValue *t, TValue *key, TValue *val, const TValue *slot);
