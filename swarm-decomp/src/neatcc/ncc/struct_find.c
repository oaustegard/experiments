#include <stdlib.h>
#include <string.h>
#include "sdef.h"
#include "ncc_data.h"
#include "ncc_data_typedefs.h"

static int struct_find(char *name, int is_union)
{
	int n = structs_n;
	int i;
	struct sdef *s;
	char *p;

	for (i = n - 1; i >= 0; i--) {
		s = (struct sdef *)structs + i;
		if (s->name[0] && strcmp(name, s->name) == 0 && s->pad[0] == is_union)
			return i;
	}
	int sz0 = structs_sz;
	if (n >= sz0) {
		int sz = structs_sz * 2;

		if (sz < 0x80)
			sz = 0x80;
		structs_sz = sz;
		p = malloc((long)sz * 0x9090);
		memcpy(p, structs, (long)n * 0x9090);
		memset(p + (long)n * 0x9090, 0, ((long)sz - n) * 0x9090);
		free(structs);
		structs = p;
	}
	structs_n = n + 1;
	s = (struct sdef *)structs + n;
	memset(s, 0, 0x9090);
	strcpy(s->name, name);
	s->pad[0] = is_union;
	return n;
}
