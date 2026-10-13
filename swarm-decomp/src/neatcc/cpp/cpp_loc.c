#include "cpp_protos.h"
#include "cpp_state.h"

char *cpp_loc(char *p)
{
	static char loc[256];
	char *end;
	int line = 1;
	int k = bufs_n - 1;

	while (k > 0 && bufs[k].kind != 0)
		k--;
	if (k != bufs_n - 1 || (long)hunk_off > (long)p) {
		end = bufs[k].buf + (int)bufs[k].cur;
		p = bufs[k].buf;
	} else {
		end = buf + (int)(cur - (hunk_off + hunk_len) + (int)(long)p);
		p = buf;
	}
	while ((p = strchr(p, '\n')) != 0 && p < end) {
		line++;
		p++;
	}
	sprintf(loc, "%s:%d", bufs[k].rest, line);
	return loc;
}
