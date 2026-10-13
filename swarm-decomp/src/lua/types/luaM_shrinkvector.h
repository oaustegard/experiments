#pragma once
#include "fn_luaM_shrinkvector_.h"

#define luaM_shrinkvector(L,v,size,fs,t) \
  ((v) = (void *)luaM_shrinkvector_(L, (v), &(size), fs, sizeof(t)))
