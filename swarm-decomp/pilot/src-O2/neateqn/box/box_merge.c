void box_merge(struct box *a, struct box *b, int c)
{
	if (box_empty(b))
		return;
	box_beforeput(a, b->f20, c);
	box_toreg(a);
	box_put(a, box_toreg(b));
	if (!a->f20)
		a->f20 = b->f20;
	if (b->cnt == 1)
		box_afterput(a, b->f24);
	else {
		a->f24 = b->f24;
		a->cnt += b->cnt;
	}
}
