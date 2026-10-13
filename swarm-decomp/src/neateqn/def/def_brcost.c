int def_brcost(int type)
{
	int i;

	for (i = 0; i < brcost_n; i++)
		if (brcost_type[i] == type && brcost_cost[i] > 0)
			return brcost_cost[i];
	return 100000;
}
