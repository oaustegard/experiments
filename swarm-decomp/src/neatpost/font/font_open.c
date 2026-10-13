#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "font.h"
#include "fn_dict_make.h"
#include "fn_font_open.h"

struct font *font_open(const char *path)
{
	FILE *fp;
	struct font *f;
	char kw[128];
	int code = -1;
	int nn = 0;
	int c;

	fp = fopen(path, "r");
	if (fp == NULL)
		return NULL;
	f = calloc(sizeof(struct font), 1);
	if (f == NULL) {
		fclose(fp);
		return NULL;
	}
	snprintf(f->desc, 0x400, "%s", path);
	f->dict898 = dict_make(-1, 1, 0);
	f->dict8a0 = dict_make(-1, 1, 0);
	f->dict8a8 = dict_make(-1, 1, 0);
	while (fscanf(fp, "%127s", kw) == 1) {
		if (strcmp(kw, "char") == 0) {
			font_readchar(f, fp, &nn, &code);
		} else if (strcmp(kw, "spacewidth") == 0) {
			fscanf(fp, "%d", &f->scale);
		} else if (strcmp(kw, "name") == 0) {
			fscanf(fp, "%s", f->pad_000);
		} else if (strcmp(kw, "fontname") == 0) {
			fscanf(fp, "%s", f->name);
		} else if (strcmp(kw, "fontpath") == 0) {
			do
				c = fgetc(fp);
			while (c == ' ');
			ungetc(c, fp);
			tilleol(fp, f->path);
		} else if (strcmp(kw, "ligatures") == 0) {
			while (fscanf(fp, "%s", kw) == 1) {
				if (strcmp(kw, "0") == 0)
					break;
			}
		} else if (strcmp(kw, "charset") == 0) {
			while (font_readchar(f, fp, &nn, &code) == 0)
				;
			break;
		}
		while ((c = getc(fp)) != '\n' && c != -1)
			;
	}
	fclose(fp);
	return f;
}
