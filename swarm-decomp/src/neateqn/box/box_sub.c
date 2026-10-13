void box_sub(struct box *b, struct box *sub, struct box *sup)
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
	int n13 = nregmk();

	if (sub)
		box_italiccorrection(sub);
	if (sup)
		box_italiccorrection(sup);
	if (sub)
		printf(".nr %s 0\\w'%s'\n", nregname(n2), box_toreg(b));
	box_italiccorrection(b);
	printf(".ps %s\n", nreg(b->type));
	tok_dim(box_toreg(b), n1, n4, n3);
	box_putf(b, "\\h'5m/100u'");
	if (sup) {
		printf(".nr %s 0\\w'%s'\n", nregname(n6), box_toreg(sup));
		if (n8)
			printf(".nr %s 0\\n[bblly]\n", nregname(n8));
		printf(".nr %s 0%su-(%dm/100u)\n", nregname(n10), nreg(n4), e_supdrop);
		printf(".if %s<(%dm/100u) .nr %s (%dm/100u)\n", nreg(n10),
		       (b->col & 1) ? e_sup3 : (b->col ? e_sup2 : e_sup1), nregname(n10),
		       (b->col & 1) ? e_sup3 : (b->col ? e_sup2 : e_sup1));
		printf(".if %s<(%s+(%dm/100u/4)) .nr %s 0%s+(%dm/100u/4)\n", nreg(n10), nreg(n8), e_xheight,
		       nregname(n10), nreg(n8), e_xheight);
	}
	if (sub) {
		printf(".nr %s 0\\w'%s'\n", nregname(n5), box_toreg(sub));
		if (n9)
			printf(".nr %s 0-\\n[bbury]\n", nregname(n9));
		printf(".nr %s 0%su+(%dm/100u)\n", nregname(n11), nreg(n3), e_subdrop);
		if (sup) {
			printf(".if %s<(%dm/100u) .nr %s (%dm/100u)\n", nreg(n11), e_sub2, nregname(n11), e_sub2);
			printf(".if (%s-%s)-(%s-%s)<(%dm/100u*4) \\{\\\n", nreg(n10), nreg(n8), nreg(n9), nreg(n11),
			       e_rulethickness);
			printf(".nr %s (%dm/100u*4)+%s-(%s-%s)\n", nregname(n11), e_rulethickness, nreg(n9), nreg(n10),
			       nreg(n8));
			printf(".nr %s (%dm/100u*4/5)-(%s-%s)\n", nregname(n12), e_xheight, nreg(n10), nreg(n8));
			printf(".if %s>0 .nr %s +%s\n", nreg(n12), nregname(n10), nreg(n12));
			printf(".if %s>0 .nr %s -%s \\}\n", nreg(n12), nregname(n11), nreg(n12));
			box_putf(b, "\\v'-%su'%s\\v'%su'", nreg(n10), box_toreg(sup), nreg(n10));
			box_putf(b, "\\h'-%su'", nreg(n6));
		} else {
			printf(".if %s<(%dm/100u) .nr %s (%dm/100u)\n", nreg(n11), e_sub1, nregname(n11), e_sub1);
			printf(".if %s<(%s-(%dm/100u*4/5)) .nr %s 0%s-(%dm/100u*4/5)\n", nreg(n11), nreg(n9), e_xheight,
			       nregname(n11), nreg(n9), e_xheight);
		}
		printf(".nr %s (%s-%s)\n", nregname(n13), nreg(n1), nreg(n2));
		printf(".if %s>0 .nr %s (%s+%s)*(%s-%s)/%s\n", nreg(n4), nregname(n13), nreg(n4), nreg(n11),
		       nreg(n1), nreg(n4), nreg(n2));
		printf(".nr %s -%s\n", nregname(n5), nreg(n13));
		box_putf(b, "\\h'-%su'", nreg(n13));
		box_putf(b, "\\v'%su'%s\\v'-%su'", nreg(n11), box_toreg(sub), nreg(n11));
		if (sup) {
			box_putf(b, "\\h'-%su'", nreg(n5));
			roff_max(n7, n5, n6);
			box_putf(b, "\\h'+%su'", nreg(n7));
		}
	} else if (sup) {
		box_putf(b, "\\v'-%su'%s\\v'%su'", nreg(n10), box_toreg(sup), nreg(n10));
	}
	box_putf(b, "\\h'%dm/100u'", e_scriptspace);
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
	nregrm(n13);
}
