static void box_colinit(struct box **items, int n, int (*arr)[4], int ra, int rb)
{
	int i, j;

	if (n == 0) {
		printf(".nr %s 0%s\n", nregname(ra), nreg(arr[0][0]));
		printf(".nr %s 0%s\n", nregname(rb), nreg(arr[0][2]));
	} else {
		for (i = 0; i < n; i++) {
			if (items[i])
				box_italiccorrection(items[i]);
		}
		for (i = 0; i < n; i++) {
			const char *s = items[i] ? box_toreg(items[i]) : "";
			int k;

			for (k = 0; k < 4; k++)
				arr[i][k] = nregmk();
			tok_len(s, arr[i][0], arr[i][1], arr[i][2], arr[i][3]);
		}
		printf(".nr %s 0%s\n", nregname(ra), nreg(arr[0][0]));
		printf(".nr %s 0%s\n", nregname(rb), nreg(arr[0][2]));
		for (i = 1; i < n; i++)
			printf(".if %s>%s .nr %s 0+%s\n", nreg(arr[i][0]), nreg(ra), nregname(ra), nreg(arr[i][0]));
		for (i = 0; i < n - 1; i++)
			printf(".if %s+%s>%s .nr %s 0+%s+%s\n", nreg(arr[i][3]), nreg(arr[i + 1][2]), nreg(rb),
			       nregname(rb), nreg(arr[i][3]), nreg(arr[i + 1][2]));
	}
	printf(".if %s>%s .nr %s 0+%s\n", nreg(arr[n - 1][3]), nreg(rb), nregname(rb), nreg(arr[n - 1][3]));
}
