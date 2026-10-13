#pragma once
#include "lstate.h"
#include "TString.h"

static TString *createstrobj (lua_State *L, size_t l, int tag, unsigned int h);
