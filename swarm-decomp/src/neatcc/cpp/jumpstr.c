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
		cur = i + 1;
		return 0;
	} else if (buf[cur] == '"') {
		i = cur;
		while (++i < len) {
			if (buf[i] == '"')
				break;
			if (buf[i] == '\\')
				i++;
		}
		cur = i + 1;
		return 0;
	}
	return 1;
}
