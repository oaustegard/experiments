#include "cpp_macro_protos.h"

static int jumpws(void)
{
	int s = cur;
	char skipped = 0;

	while (cur < len && isspace(buf[cur])) {
		cur++;
		skipped = 1;
	}
	return s == cur;
}
