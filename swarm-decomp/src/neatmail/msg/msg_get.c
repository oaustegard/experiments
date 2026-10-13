#include <stddef.h>
#include "startswith.h"

char *hdrnext(char *p, char *end);

char *msg_get(char *s, size_t n, char *key) {
	char *end = s + n;
	char *p = s;

	if (!p || p >= end)
		return NULL;
	while (p && p < end) {
		if (startswith(p, key))
			return p;
		p = hdrnext(p, end);
	}
	return NULL;
}
