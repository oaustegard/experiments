void box_from(struct box *b, struct box *m, struct box *lo, struct box *hi)
{
	int n0 = nregmk();
	int n1 = nregmk();
	int n2 = nregmk();
	int n3 = nregmk();
	int n4 = nregmk();
	int n5 = nregmk();
	int n6 = nregmk();
	int n7 = nregmk();
	int n8 = nregmk();
	int n9 = nregmk();

	box_italiccorrection(m);
	box_beforeput(b, 0x20, 0);
	tok_dim(box_toreg(m), n0, n1, n2);
	printf(".ps %s\n", nreg(b->type));
	if (hi) {
		tok_dim(box_toreg(hi), n4, 0, n5);
		if (lo) {
			tok_dim(box_toreg(lo), n3, n6, 0);
			roff_max(n9, n3, n4);
		} else {
			printf(".nr %s %s\n", nregname(n9), nreg(n4));
		}
	} else {
		if (lo)
			tok_dim(box_toreg(lo), n3, n6, 0);
		printf(".nr %s %s\n", nregname(n9), nreg(n3));
	}
	printf(".if %s>%s .nr %s 0%s\n", nreg(n0), nreg(n9), nregname(n9), nreg(n0));
	box_putf(b, "\\h'%su-%su/2u'", nreg(n9), nreg(n0));
	box_merge(b, m, 0);
	box_putf(b, "\\h'-%su/2u'", nreg(n0));
	if (hi) {
		printf(".nr %s (%dm/100u)-%s\n", nregname(n7), e_bigopspacing3, nreg(n5));
		printf(".if %s<(%dm/100u) .nr %s (%dm/100u)\n", nreg(n7), e_bigopspacing1, nregname(n7), e_bigopspacing1);
		printf(".nr %s +%s+%s\n", nregname(n7), nreg(n1), nreg(n5));
		box_putf(b, "\\h'-%su/2u'\\v'-%su'%s\\v'%su'\\h'-%su/2u'", nreg(n4), nreg(n7), box_toreg(hi), nreg(n7), nreg(n4));
	}
	if (lo) {
		printf(".nr %s (%dm/100u)-%s\n", nregname(n8), e_bigopspacing4, nreg(n6));
		printf(".if %s<(%dm/100u) .nr %s (%dm/100u)\n", nreg(n8), e_bigopspacing2, nregname(n8), e_bigopspacing2);
		printf(".nr %s +%s+%s\n", nregname(n8), nreg(n2), nreg(n6));
		box_putf(b, "\\h'-%su/2u'\\v'%su'%s\\v'-%su'\\h'-%su/2u'", nreg(n3), nreg(n8), box_toreg(lo), nreg(n8), nreg(n3));
	}
	box_putf(b, "\\h'%su/2u'", nreg(n9));
	box_afterput(b, 0x20);
	nregrm(n0);
	nregrm(n1);
	nregrm(n2);
	nregrm(n3);
	nregrm(n4);
	nregrm(n5);
	nregrm(n6);
	nregrm(n7);
	nregrm(n8);
	nregrm(n9);
}
