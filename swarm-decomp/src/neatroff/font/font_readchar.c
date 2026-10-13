static int font_readchar(struct font *f, FILE *fp, int *a, int *b)
{
	char w1[32];
	char w3[32];
	char w2[128];
	int c;
	int code;
	int n;
	struct fglyph *g;

	if (fscanf(fp, "%31s %127s", w1, w2) != 2)
		return 1;
	if (strcmp(w1, "---") == 0)
		sprintf(w1, "c%04d", *a);
	if (strcmp(w2, "\"") == 0) {
		dict_put(f->d88, w1, *b);
		return 0;
	}
	if (fscanf(fp, "%d %31s", &c, w3) != 2)
		return 1;
	code = c;
	if (f->nglyph == f->pad74) {
		f->pad74 = f->nglyph + 0x400;
		f->glyph = mextend(f->glyph, f->nglyph, f->pad74, sizeof(struct fglyph));
	}
	g = f->glyph + f->nglyph;
	snprintf((char *)g, 32, "%s", w3);
	snprintf((char *)g + 0x20, 32, "%s", w1);
	*(short *)((char *)g + 0x52) = code;
	*(struct font **)((char *)g + 0x40) = f;
	dict_put(f->d78, (char *)g, f->nglyph);
	n = f->nglyph;
	f->nglyph = n + 1;
	*b = n;
	g = f->glyph + n;
	sscanf(w2, "%hd,%hd,%hd,%hd,%hd", (short *)((char *)g + 0x48), (short *)((char *)g + 0x4a),
	       (short *)((char *)g + 0x4c), (short *)((char *)g + 0x4e), (short *)((char *)g + 0x50));
	dict_put(f->d80, w1, *b);
	(*a)++;
	return 0;
}
