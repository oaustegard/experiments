#pragma once
#include "fglyph.h"
#include "font.h"

struct fglyph *font_find(struct font *f, const char *name);
