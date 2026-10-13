#pragma once
#include "lobject.h"
#include "lstate.h"
#include "Table.h"

const TValue *luaH_get(Table *t, const TValue *key);
