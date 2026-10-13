#include <stdio.h>
#include <string.h>
#include "dev_res.h"
#include "dev_uwid.h"
#include "dev_hor.h"
#include "dev_vert.h"

int dev_open(const char *dir, const char *dev)
{
	FILE *f;
	char word[128];
	char buf[1024];
	int c, i;

	strcpy(dev_dir, dir);
	strcpy(dev_dev, dev);
	sprintf(buf, "%s/dev%s/DESC", dir, dev);
	f = fopen(buf, "r");
	if (!f)
		return 1;
	while (fscanf(f, "%127s", word) == 1) {
		if (word[0] == '#') {
			for (;;) {
				c = getc(f);
				if (c == '\n' || c == EOF)
					break;
			}
		} else if (strcmp(word, "fonts") == 0) {
			fscanf(f, "%d", &fn_n);
			for (i = 0; i < fn_n; i++)
				fscanf(f, "%s", fn_name[i + 1]);
			fn_n++;
		} else if (strcmp(word, "sizes") == 0) {
			while (fscanf(f, "%127s", word) == 1 && strcmp(word, "0") != 0)
				;
		} else if (strcmp(word, "res") == 0) {
			fscanf(f, "%d", &dev_res);
		} else if (strcmp(word, "unitwidth") == 0) {
			fscanf(f, "%d", &dev_uwid);
		} else if (strcmp(word, "hor") == 0) {
			fscanf(f, "%d", &dev_hor);
		} else if (strcmp(word, "vert") == 0) {
			fscanf(f, "%d", &dev_vert);
		} else if (strcmp(word, "charset") == 0) {
			break;
		} else {
			for (;;) {
				c = getc(f);
				if (c == '\n' || c == EOF)
					break;
			}
		}
	}
	fclose(f);
	return 0;
}
