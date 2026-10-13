#pragma once
#include "checktab.h"

#define TAB_R	1
#define TAB_W	2
#define TAB_L	4
#define TAB_RW	(TAB_R | TAB_W)

#define aux_getn(L,n,w)	(checktab(L, n, (w) | TAB_L), luaL_len(L, n))
