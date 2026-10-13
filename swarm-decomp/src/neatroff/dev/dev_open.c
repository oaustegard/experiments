#include "libc.h"
#include "fn_out.h"
#include "fn_errmsg.h"
#include "fn_dev_mnt.h"

int dev_open(char *dir, char *dev)
{
	char word[128];
	char path[1024];
	FILE *f;
	int c, i;

	snprintf(dev_dir, 1024, "%s", dir);
	snprintf(dev_dev, 1024, "%s", dev);
	snprintf(path, 1024, "%s/dev%s/DESC", dir, dev);
	f = fopen(path, "r");
	if (f == NULL)
		return 1;
	while (fscanf(f, "%127s", word) == 1) {
		if (word[0] == '#') {
			do {
				c = getc(f);
			} while (c != '\n' && c != EOF);
			continue;
		}
		if (strcmp(word, "fonts") == 0) {
			fscanf(f, "%d", &fn_n);
			for (i = 0; i < fn_n; i++)
				fscanf(f, "%s", fn_name[i + 1]);
			fn_n++;
			continue;
		}
		if (strcmp(word, "sizes") == 0) {
			do {
				if (fscanf(f, "%127s", word) != 1)
					break;
			} while (strcmp(word, "0") != 0);
			continue;
		}
		if (strcmp(word, "res") == 0) {
			fscanf(f, "%d", &dev_res);
			continue;
		}
		if (strcmp(word, "unitwidth") == 0) {
			fscanf(f, "%d", &dev_uwid);
			continue;
		}
		if (strcmp(word, "hor") == 0) {
			fscanf(f, "%d", &dev_hor);
			continue;
		}
		if (strcmp(word, "vert") == 0) {
			fscanf(f, "%d", &dev_vert);
			continue;
		}
		if (strcmp(word, "charset") == 0)
			break;
		do {
			c = getc(f);
		} while (c != '\n' && c != EOF);
	}
	fclose(f);
	out("x T %s\n", dev_dev);
	out("x res %d %d %d\n", dev_res, dev_hor, dev_vert);
	out("x init\n");
	for (i = 0; i < fn_n; i++) {
		if (fn_name[i][0] != 0) {
			if (dev_mnt(i, fn_name[i], fn_name[i]) < 0) {
				errmsg("neatroff: font %s is missing\n", fn_name[i]);
				return 1;
			}
		}
	}
	return 0;
}
