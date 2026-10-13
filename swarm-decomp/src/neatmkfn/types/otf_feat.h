#pragma once

/* one entry of a feature table: name, string, tag and lookup index */
struct otf_feat {
	char name[8];
	char str[8];
	char tag[8];
	int idx;
};
