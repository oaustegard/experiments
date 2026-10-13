void box_sqrt(struct box *b, struct box *c)
{
	char *sz[8] = {0};
	char *p0, *p1, *p2, *p3;
	int a1[4], a2[4], a3[4], a4[4];
	int s1 = sregmk();
	int n1 = nregmk();
	int n2 = nregmk();
	int n3, n4, n5, n6, n7, s2, rt, xh, i;
	char *t;

	box_italiccorrection(c);
	box_beforeput(b, 0x10, 0);
	t = box_toreg(c);
	for (i = 0; i < 4; i++)
		a1[i] = nregmk();
	tok_len(t, a1[0], a1[1], a1[2], a1[3]);
	printf(".ps %s\n", nreg(b->type));
	rt = e_rulethickness;
	xh = b->col > 1 ? rt : e_xheight;
	printf(".nr %s 0%s+%s+(2*%dm/100u)+(%dm/100u/4)\n", nregname(n2), nreg(a1[2]), nreg(a1[3]), rt, xh);
	n3 = nregmk();
	n4 = nregmk();
	n5 = nregmk();
	n6 = nregmk();
	n7 = nregmk();
	s2 = sregmk();
	printf(".nr %s 0%s/2*11/10\n", nregname(n7), nreg(n2));
	printf(".ds %s \"\n", sregname(s2));
	p0 = 0;
	p1 = 0;
	p2 = 0;
	def_pieces("\\(sr", &p0, &p1, &p2, &p3);
	def_sizes("\\(sr", sz);
	box_bracketsel(s2, n7, n7, sz, 0, 0);
	if (p1) {
		printf(".if \\w'%s' ", p1);
		printf(".if '%s'' \\{\\\n", sreg(s2));
		box_bracketmk(s2, n7, p0, p1, p2, 0);
		puts(".  \\}");
	}
	printf(".if '%s'' \\{\\\n", sreg(s2));
	for (i = 0; i < 4; i++)
		a2[i] = nregmk();
	tok_len("\\(sr", a2[0], a2[1], a2[2], a2[3]);
	printf(".ie %s<(%s+%s) .nr %s 0\\n(.s\n", nreg(n2), nreg(a2[2]), nreg(a2[3]), nregname(n3));
	printf(".el .nr %s 0%s*\\n(.s/(%s+%s-(%dm/100u))+1\n", nregname(n3), nreg(n2), nreg(a2[2]), nreg(a2[3]), rt);
	printf(".ps %s\n", nreg(n3));
	printf(".ds %s \"\\(sr\n", sregname(s2));
	for (i = 0; i < 4; i++)
		nregrm(a2[i]);
	puts(".  \\}");
	t = sreg(s2);
	for (i = 0; i < 4; i++)
		a2[i] = nregmk();
	tok_len(t, a2[0], a2[1], a2[2], a2[3]);
	printf(".nr %s \\n[bburx]\n", nregname(n5));
	for (i = 0; i < 4; i++)
		a3[i] = nregmk();
	tok_len("\\(rn", a3[0], a3[1], a3[2], a3[3]);
	printf(".nr %s 0%s-\\n[bbllx]-(%dm/100u)\n", nregname(n6), nreg(n5), rt);
	printf(".nr %s 0\n", nregname(n4));
	printf(".if %s<%s .nr %s 0%s-%s\n", nreg(a1[0]), nreg(a3[0]), nregname(n4), nreg(a3[0]), nreg(a1[0]));
	printf(".ds %s \"\\s[\\n(.s]\\f[\\n(.f]\\v'%su'\\h'%su'\\l'%su+%su\\(rn'\\h'-%su'\\v'-%su'\\h'-%su-%su'\\v'%su'%s\\v'-%su'\\h'%su+%su'\n",
	       sregname(s1), nreg(a1[0]), nreg(a1[1]), nreg(a2[0]), nreg(a2[1]), nreg(a3[0]), nreg(a3[1]),
	       nreg(a3[2]), nreg(a3[3]), nreg(a4[0]), sreg(s2), nreg(a4[1]), nreg(a4[2]), nreg(a4[3]));
	for (i = 0; i < 4; i++)
		nregrm(a2[i]);
	for (i = 0; i < 4; i++)
		nregrm(a3[i]);
	nregrm(n3);
	nregrm(n4);
	nregrm(n5);
	nregrm(n6);
	sregrm(s2);
	t = sreg(s1);
	for (i = 0; i < 4; i++)
		a4[i] = nregmk();
	tok_len(t, a4[0], a4[1], a4[2], a4[3]);
	rt = e_rulethickness;
	xh = b->col > 1 ? rt : e_xheight;
	printf(".nr %s 0(%dm/100u)+(%dm/100u/4)\n", nregname(n1), rt, xh);
	printf(".if %s>(%s+%s+%s) .nr %s (%s+%s-%s-%s)/2\n", nreg(a4[3]), nreg(a1[0]), nreg(a1[1]),
	       nreg(a1[2]), nregname(n1), nreg(a1[3]), nreg(a4[0]), nreg(a4[1]), nreg(a4[2]));
	printf(".nr %s +%s\n", nregname(n1), nreg(a2[2]));
	t = box_toreg(c);
	box_putf(b, "\\v'-%su'%s\\v'%su'\\h'-%su'%s", nreg(n1), sreg(s1), nreg(n1), nreg(a1[3]), t);
	box_afterput(b, 0x10);
	box_toreg(b);
	for (i = 0; i < 4; i++)
		nregrm(a1[i]);
	for (i = 0; i < 4; i++)
		nregrm(a4[i]);
	sregrm(s1);
	nregrm(n1);
	nregrm(n2);
}
