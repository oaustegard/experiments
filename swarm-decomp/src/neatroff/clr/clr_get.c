#include <ctype.h>
#include "clr_color.h"

int clr_get(char *s)
{
	unsigned char c = *s;
	char *p;
	size_t n;
	int i, r, g, b;
	if (c == '#') {
		n = strlen(s);
		if (n == 7) {
			r = (p = strchr("0123456789abcdef", tolower(s[1]))) ? p - "0123456789abcdef" : 0;
			if ((p = strchr("0123456789abcdef", tolower(s[2]))))
				r = r * 16 + (p - "0123456789abcdef");
			r <<= 16;
			g = (p = strchr("0123456789abcdef", tolower(s[3]))) ? p - "0123456789abcdef" : 0;
			if ((p = strchr("0123456789abcdef", tolower(s[4]))))
				g = g * 16 + (p - "0123456789abcdef");
			g = (g << 8) | r;
			r = (p = strchr("0123456789abcdef", tolower(s[5]))) ? p - "0123456789abcdef" : 0;
			if ((p = strchr("0123456789abcdef", tolower(s[6]))))
				r = r * 16 + (p - "0123456789abcdef");
			return g | r;
		} else if (n == 3) {
			r = 0;
			if ((p = strchr("0123456789abcdef", tolower(s[1]))))
				r = p - "0123456789abcdef";
			if ((p = strchr("0123456789abcdef", tolower(s[2]))))
				r = r * 16 + (p - "0123456789abcdef");
			return (r << 16) | (r << 8);
		} else if (n == 3) {
			r = 0;
			if ((p = strchr("0123456789abcdef", tolower(s[1]))))
				r = p - "0123456789abcdef";
			if ((p = strchr("0123456789abcdef", tolower(s[2]))))
				r = r * 16 + (p - "0123456789abcdef");
			return (r << 16) | (r << 8);
		} else if (n == 4) {
			r = 0;
			if ((p = strchr("0123456789abcdef", tolower(s[1])))) {
				i = p - "0123456789abcdef";
				i = i * 16 + i;
				r = i << 16;
			}
			if ((p = strchr("0123456789abcdef", tolower(s[2])))) {
				i = p - "0123456789abcdef";
				i = i * 16 + i;
				r |= i << 8;
			}
			if ((p = strchr("0123456789abcdef", tolower(s[3])))) {
				i = p - "0123456789abcdef";
				i = i * 16 + i;
				r |= i;
			}
			return r;
		} else if (n == 2) {
			if (!(p = strchr("0123456789abcdef", tolower(s[1]))))
				return 0;
			i = p - "0123456789abcdef";
			i = i * 16 + i;
			return (i << 16) | (i << 8) | i;
		}
	}
	if (isdigit(c)) {
		if ((int)strtol(s, NULL, 10) >= 0 && (unsigned)(int)strtol(s, NULL, 10) <= 7)
			return colors[(int)strtol(s, NULL, 10)].rgb;
	}
	for (i = 0; i < 8; i++)
		if (strcmp(colors[i].name, s) == 0)
			return colors[i].rgb;
	return 0;
}
