#include "cpp_macro_protos.h"

static int jumpstr(void)
{
	long i;

	if (buf[cur] == '\'') {
		i = cur;
		while (++i < len) {
			if (buf[i] == '\'')
				break;
			if (buf[i] == '\\')
				i++;
		}
	} else if (buf[cur] == '"') {
		i = cur;
		while (++i < len) {
			if (buf[i] == '"')
				break;
			if (buf[i] == '\\')
				i++;
		}
	} else
		return 1;
	cur = i + 1;
	return 0;
}
