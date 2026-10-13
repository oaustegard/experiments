void box_bar(struct box *b)
{
	int r14 = nregmk();
	int rbx = nregmk();
	int r15 = nregmk();
	int r13 = nregmk();
	int r12 = nregmk();

	box_italiccorrection(b);
	printf(".ps %s\n", nreg(b->type));
	printf(".nr %s 0\\w'%s'\n", nregname(r15), "\\(ru");
	if (r13)
		printf(".nr %s 0\\n[bblly]-1\n", nregname(r13));
	printf(".nr %s 0\\w'%s'\n", nregname(r14), box_toreg(b));
	if (rbx)
		printf(".nr %s 0-\\n[bbury]\n", nregname(rbx));
	printf(".if %su<(%dm/100u) .nr %s 0%dm/100u\n", nreg(rbx), e_xheight, nregname(rbx), e_xheight);
	printf(".nr %s 0%su+%su+(3*%dm/100u)\n", nregname(r12), nreg(rbx), nreg(r13), e_rulethickness);
	box_putf(b, "\\v'-%su'\\s%s\\f[\\n(.f]\\l'-%su\\(ru'\\v'%su'", nreg(r12), escarg(nreg(b->type)), nreg(r14), nreg(r12));
	nregrm(r14);
	nregrm(rbx);
	nregrm(r15);
	nregrm(r13);
	nregrm(r12);
}
