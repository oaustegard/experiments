void box_accent(struct box *b, const char *s)
{
	int r14 = nregmk();
	int rbp = nregmk();
	int r13 = nregmk();
	int r12 = nregmk();
	int r15 = nregmk();

	box_italiccorrection(b);
	printf(".ps %s\n", nreg(b->type));
	printf(".nr %s 0\\w'%s'\n", nregname(r12), s);
	if (r15)
		printf(".nr %s 0\\n[bblly]-1\n", nregname(r15));
	printf(".nr %s 0\\w'%s'\n", nregname(r14), box_toreg(b));
	if (rbp)
		printf(".nr %s 0-\\n[bbury]\n", nregname(rbp));
	printf(".if %su<(%dm/100u) .nr %s 0%dm/100u\n", nreg(rbp), e_xheight, nregname(rbp), e_xheight);
	printf(".nr %s 0%su+%su+(%sp*10u/100u)\n", nregname(r13), nreg(rbp), nreg(r15), nreg(b->type));
	box_putf(b, "\\v'-%su'\\h'-%su-%su/2u'\\s%s\\f[\\n(.f]%s\\h'%su-%su/2u'\\v'%su'", nreg(r13), nreg(r14), nreg(r12), escarg(nreg(b->type)), s, nreg(r14), nreg(r12), nreg(r13));
	nregrm(r14);
	nregrm(rbp);
	nregrm(r13);
	nregrm(r12);
	nregrm(r15);
}
