void box_vertspace(struct box *b)
{
	int r13 = nregmk();
	int r12 = nregmk();
	int bp = nregmk();

	box_italiccorrection(b);
	printf(".nr %s 0+\\n(.vu-%sp+(%sp*%du/100u)\n", nregname(r12), nreg(b->type), nreg(b->type), e_bodyheight);
	printf(".nr %s 0+\\n(.vu-%sp+(%sp*%du/100u)\n", nregname(bp), nreg(b->type), nreg(b->type), e_bodydepth);
	printf(".nr %s 0\\w'%s'\n", nregname(r13), box_toreg(b));
	printf(".if -\\n[bbury]>%s .as %s \"\\x'\\n[bbury]u+%su'\n", nreg(r12), sregname(b->sreg_id), nreg(r12));
	printf(".if \\n[bblly]>%s .as %s \"\\x'\\n[bblly]u-%su'\n", nreg(bp), sregname(b->sreg_id), nreg(bp));
	nregrm(r13);
	nregrm(r12);
	nregrm(bp);
}
