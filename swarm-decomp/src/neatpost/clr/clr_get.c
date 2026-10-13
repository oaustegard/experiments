#include <ctype.h>
#include "clr_color.h"
#include "sys_std.h"

int
clr_get(const char *s)
{
	const char *hex = "0123456789abcdef";
	const char *p;
	int r, i;

	if (*s == '#') {
		size_t n = strlen(s);
		if (n == 7) {
			r = (p = strchr(hex, tolower(s[1]))) ? p - hex : 0;
			if (p = strchr(hex, tolower(s[2])))
				r = r * 16 + (p - hex);
			r <<= 16;
			i = (p = strchr(hex, tolower(s[3]))) ? p - hex : 0;
			if (p = strchr(hex, tolower(s[4])))
				i = i * 16 + (p - hex);
			i <<= 8;
			r |= i;
			i = (p = strchr(hex, tolower(s[5]))) ? p - hex : 0;
			if (p = strchr(hex, tolower(s[6])))
				i = i * 16 + (p - hex);
			return r | i;
		}
		if (n == 4) {
			r = 0;
			p = strchr(hex, tolower(s[1]));
			if (p) r = (p - hex) * 17 << 16;
			p = strchr(hex, tolower(s[2]));
			if (p) r |= (p - hex) * 17 << 8;
			p = strchr(hex, tolower(s[3]));
			if (p) r |= (p - hex) * 17;
			return r;
		}
	}
	if (isdigit(*s)) {
		if ((int)strtol(s, 0, 10) >= 0 && (unsigned)strtol(s, 0, 10) <= 7)
			return colors[(int)strtol(s, 0, 10)].val;
	}
	for (i = 0; i < 8; i++)
		if (strcmp(colors[i].name, s) == 0)
			return colors[i].val;
	return 0;
}
