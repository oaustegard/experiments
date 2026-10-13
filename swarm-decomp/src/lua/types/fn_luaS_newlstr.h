#pragma once
#include <stddef.h>
#include "lstate.h"

struct TString;
struct TString *luaS_newlstr (lua_State *L, const char *str, size_t l);
