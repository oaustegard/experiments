#include <stdio.h>
#include <string.h>
#include "font.h"
#include "fn_font_open.h"

struct font *dev_fontopen(const char *name)
{
	char buf[1024];
	if (strchr(name, '/') != NULL) {
		strcpy(buf, name);
	} else {
		sprintf(buf, "%s/dev%s/%s", dev_dir, dev_dev, name);
	}
	return font_open(buf);
}
