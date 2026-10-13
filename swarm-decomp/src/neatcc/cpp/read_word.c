#include "cpp_macro_protos.h"

static void read_word(char *d)
{
	jumpws();
	while (cur < len && (isalnum(buf[cur]) || buf[cur] == '_'))
		*d++ = buf[cur++];
	*d = 0;
}
