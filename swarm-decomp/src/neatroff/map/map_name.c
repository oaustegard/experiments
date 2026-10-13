#include "dict.h"
#include "map_dict.h"

char *map_name(int c)
{
	static volatile char map_buf[128];
	char *r;

	if (c > 0xff)
		return dict_key(mapdict, c - 0x100);
	map_buf[0] = '.';
	r = (char *)map_buf;
	map_buf[1] = c;
	map_buf[2] = 0;
	return r;
}
