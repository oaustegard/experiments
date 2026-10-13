void box_wrap(struct box *b, struct box *m, const char *l, const char *r)
{
	int arr[4];
	char *s;
	int i;

	s = box_toreg(m);
	for (i = 0; i < 4; i++)
		arr[i] = nregmk();
	tok_len(s, arr[0], arr[1], arr[2], arr[3]);
	printf(".ps %s\n", nreg(b->type));
	if (l) {
		box_beforeput(b, 0x50, 0);
		box_bracket(b, bracsign(l, 1), arr[2], arr[3]);
		box_afterput(b, 0x50);
	}
	box_merge(b, m, 0);
	if (r) {
		box_beforeput(b, 0x60, 0);
		box_bracket(b, bracsign(r, 0), arr[2], arr[3]);
		box_afterput(b, 0x60);
	}
	for (i = 0; i < 4; i++)
		nregrm(arr[i]);
}
