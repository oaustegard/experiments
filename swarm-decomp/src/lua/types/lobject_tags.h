#pragma once
#include "TValue.h"

#define LUA_VNIL 0
#define LUA_VNUMINT 3
#define BIT_ISCOLLECTABLE (1 << 6)
#define BLACKBIT 5
#define WHITEBITS (3 << 3)

#define ttypetag(o) ((o)->tt_ & 0x3f)
#define ttisinteger(o) ((o)->tt_ == LUA_VNUMINT)
#define ivalue(o) ((o)->value_.i)
#define setivalue(obj,x) { TValue *io_ = (obj); io_->value_.i = (x); io_->tt_ = LUA_VNUMINT; }
#define setnilvalue(obj) ((obj)->tt_ = LUA_VNIL)
#define setobj(obj1,obj2) { TValue *io1_ = (obj1); const TValue *io2_ = (obj2); io1_->value_ = io2_->value_; io1_->tt_ = io2_->tt_; }

/* variant tags (lobject.h of Lua 5.4) */
#define LUA_VFALSE	0x01
#define LUA_VNUMFLT	0x13
#define LUA_VSHRSTR	0x44
#define LUA_VTABLE	0x45
#define LUA_VUSERDATA	0x47

/* l_isfalse: nil or the boolean false (lobject.h of Lua 5.4) */
#define l_isfalse(o) (((o)->tt_ == LUA_VFALSE) || (((o)->tt_ & 0xf) == 0))
