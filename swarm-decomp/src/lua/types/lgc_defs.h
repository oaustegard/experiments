#pragma once
#include "lstate.h"
#include "Table.h"
#include "Node.h"
#include "UpVal.h"
#include "Proto.h"
#include "LClosure.h"

#define cast_byte(x) ((lu_byte)(x))

/* age values (lgc.h) */
#define G_NEW 0
#define G_SURVIVAL 1
#define G_OLD0 2
#define G_OLD1 3
#define G_OLD 4
#define G_TOUCHED1 5
#define G_TOUCHED2 6
#define AGEBITS 7

#define WHITE0BIT 3
#define WHITE1BIT 4
#define BLACKBIT 5
#define FINALIZEDBIT 6
#define WHITEBITS ((1 << WHITE0BIT) | (1 << WHITE1BIT))
#define maskcolors ((1 << BLACKBIT) | WHITEBITS)

#define KGC_INC 0
#define KGC_GEN 1

#define GCSpropagate 0
#define GCSenteratomic 1
#define GCSatomic 2
#define GCSswpallgc 3
#define GCSswpfinobj 4
#define GCSswptobefnz 5
#define GCSswpend 6
#define GCScallfin 7
#define GCSpause 8

#define keepinvariant(g) ((g)->gcstate <= GCSatomic)

#define getage(o) ((o)->marked & AGEBITS)
#define setage(o, a) ((o)->marked = cast_byte(((o)->marked & (~AGEBITS)) | (a)))
#define isold(o) (getage(o) > G_SURVIVAL)
#define changeage(o, f, t) ((o)->marked = cast_byte(((o)->marked & (~AGEBITS)) | (t)))

#define iswhite(x) ((x)->marked & WHITEBITS)
#define isblack(x) ((x)->marked & (1 << BLACKBIT))
#define set2gray(x) ((x)->marked &= cast_byte(~maskcolors))
#define set2black(x) ((x)->marked = cast_byte(((x)->marked & ~maskcolors) | (1 << BLACKBIT)))

#define luaC_white(g) ((g)->currentwhite & WHITEBITS)
#define makewhite(g, x) ((x)->marked = cast_byte(((x)->marked & ~maskcolors) | luaC_white(g)))

#define ttiscollectable(o) ((o)->tt_ & (1 << 6))
#define iscollectable(o) ttiscollectable(o)
#define gcvalue(o) ((o)->value_.gc)
#define markvalue(g, o) do { if (iscollectable(o) && iswhite(gcvalue(o))) reallymarkobject(g, gcvalue(o)); } while (0)
#define markobject(g, t) do { if (iswhite(t)) reallymarkobject(g, (GCObject *)(t)); } while (0)
#define markobjectN(g, t) do { if (t) markobject(g, t); } while (0)

#define lua_assert(c) ((void)0)

#define LUA_TDEADKEY 11
#define novariant(t) ((t) & 0x0F)
#define valiswhite(x) (iscollectable(x) && iswhite(gcvalue(x)))
#define isempty(v) ((v)->tt_ & 0x0F) == 0
