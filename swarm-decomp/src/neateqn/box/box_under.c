void box_under(struct box *b)
{
	int r14 = nregmk();
	int rbx = nregmk();
	int r15 = nregmk();
	int r13 = nregmk();
	int r12 = nregmk();

	box_italiccorrection(b);
	printf(".ps %s\n", nreg(b->type));
	printf(".nr %s 0\\w'%s'\n", nregname(r15), "\\(ul");
	if (r13)
		printf(".nr %s 0-\\n[bbury]-1\n", nregname(r13));
	printf(".nr %s 0\\w'%s'\n", nregname(r14), box_toreg(b));
	if (rbx)
		printf(".nr %s 0\\n[bblly]\n", nregname(rbx));
	printf(".if %s<0 .nr %s 0\n", nreg(rbx), nregname(rbx));
	printf(".nr %s 0%su+%su+(3*%dm/100u)\n", nregname(r12), nreg(rbx), nreg(r13), e_rulethickness);
	box_putf(b, "\\v'%su'\\s%s\\f[\\n(.f]\\l'-%su\\(ul'\\v'-%su'", nreg(r12), escarg(nreg(b->type)), nreg(r14), nreg(r12));
	nregrm(r14);
	nregrm(rbx);
	nregrm(r15);
	nregrm(r13);
	nregrm(r12);
}
