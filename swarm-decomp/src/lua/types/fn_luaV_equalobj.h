#pragma once
#include <lua.h>
#include "TValue.h"

int luaV_equalobj(lua_State *L, const TValue *t1, const TValue *t2);
