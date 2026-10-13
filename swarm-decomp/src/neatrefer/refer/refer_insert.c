static void refer_insert(int *ids, int n)
{
	int *end;

	if (n <= 0)
		return;
	end = ids + n;
	do {
		inserted++;
		ref_ins(cites[*ids], inserted);
		ids++;
	} while ((long)ids != (long)end);
}
