#include "dev_font.h"

void outfont(int id)
{
	if (dev_font(id))
		o_f = id;
}
