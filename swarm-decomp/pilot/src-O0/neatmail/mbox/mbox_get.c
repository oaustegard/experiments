int mbox_get(struct mbox *mbox, int i, char **beg, char **end)
{
	if (i < 0 || i >= mbox->nmail)
		return 1;
	if (mbox->cur_beg[i]) {
		*beg = mbox->cur_beg[i];
		*end = mbox->cur_end[i];
	} else {
		*beg = mbox->orig_beg[i];
		*end = mbox->orig_end[i];
	}
	return 0;
}
