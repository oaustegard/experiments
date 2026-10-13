#include <ctype.h>

int startswith(const char *s, const char *p) {
	while (*p) {
		if (tolower((unsigned char)*p++) != tolower((unsigned char)*s++))
			return 0;
	}
	return 1;
}
