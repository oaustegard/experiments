#pragma once
#include "lobject.h"
#include "LexState.h"
TString *luaX_newstring(LexState *ls, const char *str, size_t l);
