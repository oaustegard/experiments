#include <stdio.h>
#include "clr_color.h"
#include "sys_std.h"

char *
clr_str(int c)
{
	static char clr_buf[32];

	if (c == 0)
		return "0";
	sprintf(clr_buf, "#%02x%02x%02x", (c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff);
	return clr_buf;
}
