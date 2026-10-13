#pragma once
#include <stddef.h>
#include <stdint.h>
#include "lua.h"
#include "lu_byte.h"
#include "GCObject.h"
#include "Value.h"
#include "TValue.h"

typedef size_t lu_mem;
typedef ptrdiff_t l_mem;
typedef struct Table Table;

typedef TValue *StkId;
