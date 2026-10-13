#pragma once
#include "lstate.h"

static const char *funcnamefromcode (lua_State *L, const Proto *p, int pc, const char **name);
