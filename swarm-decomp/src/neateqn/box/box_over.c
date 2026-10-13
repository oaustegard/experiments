void box_over(struct box *b, struct box *num, struct box *den)
{
	int n1 = nregmk();
	int n2 = nregmk();
	int n3 = nregmk();
	int n4 = nregmk();
	int n5 = nregmk();
	int n6 = nregmk();
	int n7 = nregmk();
	int n8 = nregmk();
	int n9 = nregmk();
	int n10 = nregmk();
	int n11 = nregmk();
	int n12 = nregmk();
	int h = ((unsigned)b->col < 2 ? 7 : 3) * e_rulethickness / 2;
	int num_v, den_v;

	box_beforeput(b, 0x80, 0);
	box_italiccorrection(num);
	box_italiccorrection(den);
	printf(".nr %s 0\\w'%s'\n", nregname(n1), box_toreg(num));
	if (n2)
		printf(".nr %s 0\\n[bblly]\n", nregname(n2));
	printf(".nr %s 0\\w'%s'\n", nregname(n3), box_toreg(den));
	if (n4)
		printf(".nr %s 0-\\n[bbury]\n", nregname(n4));
	roff_max(n5, n1, n3);
	printf(".ps %s\n", nreg(b->type));
	tok_len("\\(ru", n8, 0, n10, n9);
	num_v = e_num2;
	if ((unsigned)b->col <= 1)
		num_v = e_num1;
	printf(".nr %s 0%dm/100u\n", nregname(n6), num_v);
	den_v = e_denom2;
	if ((unsigned)b->col <= 1)
		den_v = e_denom1;
	printf(".nr %s 0%dm/100u\n", nregname(n7), den_v);
	printf(".nr %s (%s-%s)-((%dm/100u)+(%dm/100u/2))\n", nregname(n12), nreg(n6), nreg(n2), e_axisheight,
	       e_rulethickness);
	printf(".if %s<(%dm/100u) .nr %s +(%dm/100u)-%s\n", nreg(n12), h, nregname(n6), h, nreg(n12));
	printf(".nr %s ((%dm/100u)-(%dm/100u/2))-(%s-%s)\n", nregname(n12), e_axisheight, e_rulethickness, nreg(n4),
	       nreg(n7));
	printf(".if %s<(%dm/100u) .nr %s +(%dm/100u)-%s\n", nreg(n12), h, nregname(n7), h, nreg(n12));
	printf(".nr %s 0-%s+%s/2-(%dm/100u)\n", nregname(n11), nreg(n9), nreg(n10), e_axisheight);
	printf(".nr %s +2*(%dm/100u)\n", nregname(n5), e_overhang);
	box_putf(b, "\\h'%sp*%du/100u'", nreg(b->type), e_nulldelim);
	box_putf(b, "\\v'%su'\\f[\\n(.f]\\s[%s]\\l'%su'\\v'-%su'\\h'-%su/2u'", nreg(n11), nreg(b->type), nreg(n5),
		 nreg(n11), nreg(n5));
	box_putf(b, "\\h'-%su/2u'", nreg(n1));
	box_putf(b, "\\v'-%su'%s\\v'%su'", nreg(n6), box_toreg(num), nreg(n6));
	box_putf(b, "\\h'-%su/2u'", nreg(n1));
	box_putf(b, "\\h'-%su/2u'", nreg(n3));
	box_putf(b, "\\v'%su'%s\\v'-%su'", nreg(n7), box_toreg(den), nreg(n7));
	box_putf(b, "\\h'(-%su+%su)/2u'", nreg(n3), nreg(n5));
	box_putf(b, "\\h'%sp*%du/100u'", nreg(b->type), e_nulldelim);
	box_afterput(b, 0x80);
	box_toreg(b);
	nregrm(n1);
	nregrm(n2);
	nregrm(n3);
	nregrm(n4);
	nregrm(n5);
	nregrm(n6);
	nregrm(n7);
	nregrm(n8);
	nregrm(n9);
	nregrm(n10);
	nregrm(n11);
	nregrm(n12);
}
