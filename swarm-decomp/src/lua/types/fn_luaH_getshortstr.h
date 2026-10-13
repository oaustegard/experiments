#pragma once
#include "lstate.h"
#include "Table.h"
#include "TString.h"

const TValue *luaH_getshortstr(Table *t, TString *key);
