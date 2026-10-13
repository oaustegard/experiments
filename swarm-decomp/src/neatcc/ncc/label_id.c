static int label_id(char *s)
{
	long i;

	if (label_n >= label_sz) {
		int sz = label_sz * 2;
		void *np, *ni;

		if (sz < 0x80)
			sz = 0x80;
		label_sz = sz;
		np = malloc((long)sz * 128);
		memcpy(np, label_name, (long)label_n * 128);
		memset((char *)np + (long)label_n * 128, 0, ((long)sz - label_n) * 128);
		free(label_name);
		label_name = np;
		ni = malloc((long)sz * 4);
		memcpy(ni, label_ids, (long)label_n * 4);
		memset((int *)ni + label_n, 0, ((long)sz - label_n) * 4);
		free(label_ids);
		label_ids = ni;
	}
	for (i = label_n - 1; i >= 0; i--) {
		if (strcmp((char *)label_name + i * 128, s) == 0)
			return ((int *)label_ids)[i];
	}
	strcpy((char *)label_name + (long)label_n * 128, s);
	((int *)label_ids)[label_n] = label + 1;
	label_n++;
	label++;
	return label;
}
