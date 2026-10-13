void def_brcostput(int type, int cost)
{
	int i, n;

	if (type == 0) {
		brcost_n = 0;
	} else {
		n = brcost_n;
		for (i = 0; i < n; i++)
			if (brcost_type[i] == type)
				break;
		if (type <= 0)
			return;
		if (i + (n <= i) >= 32)
			return;
		brcost_type[i] = type;
		brcost_cost[i] = cost;
		if (n > i)
			return;
		brcost_n = i + 1;
	}
}
