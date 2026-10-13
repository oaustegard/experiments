#include <string.h>

#define U16(p) (((p)[0] << 8) | (p)[1])
#define U32(p) (((unsigned)(p)[0] << 24) | ((p)[1] << 16) | ((p)[2] << 8) | (p)[3])

static unsigned char *
otf_table(const unsigned char *base, const unsigned char *tab, const char *tag, int unused)
{
	const unsigned char *p;
	unsigned int n;

	n = U16(tab + 4);
	if (n == 0)
		return 0;
	p = tab + 12;
	do {
		if (strncmp((const char *)p, tag, 4) == 0)
			return (unsigned char *)(base + U32(p + 8));
		p += 16;
	} while (p != tab + 28 + (size_t)(n - 1) * 16);
	return 0;
}
