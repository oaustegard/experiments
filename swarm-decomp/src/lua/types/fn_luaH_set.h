#pragma once
#include <lua.h>
#include "lobject.h"
#include "lstate.h"
#include "Table.h"

void luaH_set(lua_State *L, Table *t, const TValue *key, TValue *value);
