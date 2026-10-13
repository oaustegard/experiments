#include <stdio.h>
#include "font.h"
#include "fn_font_close.h"

struct font *dev_fontopen(const char *name);

int dev_mnt(int n, const char *name, const char *path)
{
	struct font *f;
	if (n > 1023)
		return -1;
	f = dev_fontopen(path);
	if (!f)
		return -1;
	if (fn_font[n])
		font_close(fn_font[n]);
	if (fn_name[n] != path)
		snprintf(fn_name[n], 64, "%s", name);
	fn_font[n] = f;
	return n;
}
