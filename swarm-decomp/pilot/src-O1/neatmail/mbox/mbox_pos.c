int mbox_pos(struct mbox *mbox, int pos, int *rel)
{
	int i;

	for (i = 1; i < mbox->noff; i++)
		if (pos < mbox->off[i])
			break;
	*rel = pos - mbox->off[i - 1];
	return i - 1;
}
