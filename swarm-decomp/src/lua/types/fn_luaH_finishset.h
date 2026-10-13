#pragma once
#include <lua.h>
#include "lobject.h"
#include "Table.h"

void luaH_finishset(lua_State *L, Table *t, const TValue *key, const TValue *slot, TValue *value);
