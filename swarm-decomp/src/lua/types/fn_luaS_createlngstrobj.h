#pragma once
#include <stddef.h>
#include "lstate.h"

struct TString;
struct TString *luaS_createlngstrobj (lua_State *L, size_t l);
