#include "libc.h"
#include "fn_font_open.h"
#include "fn_font_close.h"
#include "fn_out.h"

int dev_mnt(int n, char *s, char *name)
{
	char buf[1024];
	void *f;
	char *p;
	int i;

	if (n > 31)
		return -1;
	if (strchr(name, '/'))
		snprintf(buf, 1024, "%s", name);
	else
		snprintf(buf, 1024, "%s/dev%s/%s", dev_dir, dev_dev, name);
	f = font_open(buf);
	if (f == NULL)
		return -1;
	if (n < 0) {
		for (n = 1, p = fn_name[1]; n < 32; n++, p += 32)
			if (strcmp(p, s) == 0)
				break;
		if (n == 32) {
			for (i = 1; i < 32 && fn_font[i]; i++)
				;
			if (i == 32)
				i = 0;
			n = i;
		}
	}
	p = fn_name[n];
	if (fn_font[n])
		font_close(fn_font[n]);
	if (name != p)
		snprintf(p, 32, "%s", s);
	fn_font[n] = f;
	out("x font %d %s\n", n, name);
	return n;
}
