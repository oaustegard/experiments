#pragma once
#include "lobject_tags.h"

#define setfltvalue(obj, x) { TValue *io_ = (obj); io_->value_.n = (x); io_->tt_ = LUA_VNUMFLT; }
