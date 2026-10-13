void box_vcenter(struct box *b, struct box *m)
{
	int r15 = nregmk();
	int r14 = nregmk();
	int r13 = nregmk();
	int bp = nregmk();

	box_beforeput(b, m->first, 0);
	tok_dim(box_toreg(m), r15, r14, r13);
	printf(".nr %s 0-%s+%s/2-(%sp*%du/100u)\n", nregname(bp), nreg(r13), nreg(r14), nreg(b->type), e_axisheight);
	box_putf(b, "\\v'%su'%s\\v'-%su'", nreg(bp), box_toreg(m), nreg(bp));
	box_toreg(b);
	box_afterput(b, m->flags);
	nregrm(r15);
	nregrm(r14);
	nregrm(r13);
	nregrm(bp);
}
