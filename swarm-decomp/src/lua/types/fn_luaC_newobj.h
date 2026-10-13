#pragma once
#include <stddef.h>
#include "lstate.h"
#include "GCObject.h"

GCObject *luaC_newobj(lua_State *L, int tt, size_t sz);
