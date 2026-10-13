#pragma once

/* hash table: keys and values in insertion order, chained per first byte of key */
struct tab {
	char **key;	/* keys */
	void **val;	/* values */
	int n;		/* entries used */
	int *idx;	/* per-entry chain link, -1 ends */
	int hash[256];	/* first byte of key -> first entry, -1 if none */
};
