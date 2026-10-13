#pragma once
#include "lobject.h"
#include "lstate.h"

void luaV_finishget (lua_State *L, const TValue *t, TValue *key, StkId val, const TValue *slot);
