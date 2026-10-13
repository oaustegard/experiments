#pragma once
#include <stdarg.h>
#include "lstate.h"

const char *luaO_pushvfstring (lua_State *L, const char *fmt, va_list argp);
