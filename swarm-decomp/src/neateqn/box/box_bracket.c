static void box_bracket(struct box *b, const char *s, int x, int y)
{
	char *p0 = 0, *p1 = 0, *p2 = 0, *p3 = 0;
	char *sz[8] = {0};
	int arr[4];
	char *t;
	int sid = sregmk();
	int nid = nregmk();
	int r12 = nregmk();
	int i;

	roff_max(nid, x, y);
	def_sizes(s, sz);
	printf(".ds %s \"\n", sregname(sid));
	def_pieces(s, &p0, &p1, &p2, &p3);
	box_bracketsel(sid, x, y, sz, p1 == 0, 1);
	if (p1) {
		printf(".if '%s'' \\{\\\n", sreg(sid));
		box_bracketmk(sid, nid, p0, p1, p2, p3);
		puts(".  \\}");
	}
	t = sreg(sid);
	for (i = 0; i < 4; i++)
		arr[i] = nregmk();
	tok_len(t, arr[0], arr[1], arr[2], arr[3]);
	printf(".nr %s 0-%s+%s/2-(%sp*%du/100u)\n", nregname(r12), nreg(arr[3]), nreg(arr[2]), nreg(b->type), e_axisheight);
	box_putf(b, "\\f[\\n(.f]\\s[\\n(.s]\\v'%su'%s\\v'-%su'", nreg(r12), sreg(sid), nreg(r12));
	box_toreg(b);
	for (i = 0; i < 4; i++)
		nregrm(arr[i]);
	sregrm(sid);
	nregrm(nid);
	nregrm(r12);
}
