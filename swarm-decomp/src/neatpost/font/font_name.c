#include "font.h"

char *font_name(struct font *f)
{
	return f->name[0] ? f->name : (char *)f;
}
