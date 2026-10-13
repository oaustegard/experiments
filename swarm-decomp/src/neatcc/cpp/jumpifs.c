#include "cpp_macro_protos.h"
#include "cpp_state.h"

static void jumpifs(int taken)
{
	char name[128];
	int depth = 0;

	while (cur < len) {
		if (buf[cur] != '#') {
			if (jumpcomment() == 0)
				continue;
			if (jumpstr() != 0)
				cur++;
			continue;
		}
		cur++;
		read_word(name);
		if (!strcmp(name, "else") && !(taken | depth))
			return;
		if (!strcmp("elif", name) && !(taken | depth)) {
			if (cpp_eval())
				return;
		}
		if (!strcmp(name, "endif")) {
			if (depth == 0)
				return;
			depth--;
		}
		if (!strcmp(name, "ifdef") || !strcmp(name, "ifndef") || !strcmp(name, "if"))
			depth++;
	}
}
