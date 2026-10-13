#pragma once
#include "lobject.h"

/* type tags (Lua 5.4 lobject.h) */
#define LUA_VNIL 0
#define LUA_VEMPTY (LUA_VNIL | (1 << 4))
#define LUA_VNUMINT (3 | (0 << 4))
#define LUA_VNUMFLT (3 | (1 << 4))
#define LUA_VSHRSTR (4 | (0 << 4))
#define LUA_VLNGSTR (4 | (1 << 4))
#define LUA_VTABLE (5 | (0 << 4))
#define LUA_VLCL (6 | (0 << 4))
#define LUA_VLCF (6 | (1 << 4))
#define LUA_VUSERDATA (7 | (0 << 4))

#define ttypetag(o) ((o)->tt_ & 0x3f)
#define ttisnil(o) (((o)->tt_ & 0xf) == LUA_VNIL)
#define ttisstring(o) (((o)->tt_ & 0xf) == 4)
#define ttisnumber(o) (((o)->tt_ & 0xf) == 3)
#define ttisinteger(o) ((o)->tt_ == LUA_VNUMINT)
#define ttisfloat(o) ((o)->tt_ == LUA_VNUMFLT)
#define ttisstrictnil(o) ((o)->tt_ == LUA_VNIL)
#define ivalue(o) ((o)->value_.i)
#define fltvalue(o) ((o)->value_.n)
#define hvalue(o) ((Table *)(o)->value_.gc)
#define tsvalue(o) ((TString *)(o)->value_.gc)
#define setivalue(obj, x) { TValue *io_ = (obj); io_->value_.i = (x); io_->tt_ = LUA_VNUMINT; }
#define setobj(L, obj1, obj2) { TValue *io1_ = (obj1); const TValue *io2_ = (obj2); io1_->value_ = io2_->value_; io1_->tt_ = io2_->tt_; }
#define s2v(o) (o)
#define ttistable(o) ((o)->tt_ == (LUA_VTABLE | 0x40))
#define ttisfunction(o) (((o)->tt_ & 0xf) == 6)
#define setnilvalue(obj) ((obj)->tt_ = LUA_VNIL)
#define isempty(v) ttisnil(v)
