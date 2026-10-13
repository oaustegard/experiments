#include <stdlib.h>
#include "font.h"
#include "fn_dict_free.h"

void font_close(struct font *f)
{
	dict_free(f->dict898);
	dict_free(f->dict8a0);
	dict_free(f->dict8a8);
	free(f->glyphs);
	free(f);
}
