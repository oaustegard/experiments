#include "cpp_protos.h"
#include "cpp_state.h"

int cpp_read(char **buf_out, long *len_out)
{
	char name[128];
	int start;
	int end;
	int k;
	int len;

	*len_out = 0;
	*buf_out = 0;
	if (seen_macro == 1) {
		macro_expand(seen_name);
		seen_macro = 0;
	}
	if (cur == len) {
		if (bufs_limit >= bufs_n)
			return 1;
		buf_pop();
	}
	start = cur;
	if (cur < len && buf[cur] == '#') {
		if (cpp_cmd() == 0)
			return 0;
	}
	end = -1;
	while (cur < len) {
		if (jumpws() == 0)
			continue;
		if (buf[cur] == '#')
			break;
		if (jumpcomment() == 0)
			continue;
		if (seen_macro == 2) {
			if (buf[cur] == '(')
				macro_expand(seen_name);
			seen_macro = 0;
			start = cur;
			continue;
		}
		if (jumpstr() == 0)
			continue;
		if (!isalnum(buf[cur]) && buf[cur] != '_') {
			cur++;
			continue;
		}
		read_word(name);
		if (buf_arg_find(name) >= 0) {
			seen_macro = 1;
			end = cur - (int)(stpcpy(seen_name, name) - seen_name);
			break;
		}
		for (k = mhead[(unsigned char)name[0]]; k > 0; k = mnext[k])
			if (!strcmp(name, macros[k].name) && macros[k].flags == 0)
				break;
		if (k > 0) {
			seen_macro = macros[k].isfunc + 1;
			end = cur - (int)(stpcpy(seen_name, name) - seen_name);
			break;
		}
		seen_macro = 0;
		for (k = bufs_n - 1; k >= 0; k--)
			if (bufs[k].kind == 3)
				break;
		if (k >= 0 && !strcmp(name, "defined")) {
			jumpws();
			if (buf[cur] == '(') {
				cur++;
				read_word(name);
				jumpws();
				cur++;
			} else
				read_word(name);
		}
	}
	if (end < 0)
		end = cur;
	len = end - start;
	for (k = bufs_n - 1; k >= 0; k--)
		if (bufs[k].kind == 3)
			break;
	if (k < 0) {
		hunk_off += hunk_len;
		hunk_len = len;
	}
	*buf_out = buf + start;
	*len_out = len;
	return 0;
}
