#pragma once
#include "lobject.h"
#include "Table.h"
#include "TString.h"

const TValue *luaH_getstr(Table *t, TString *key);
