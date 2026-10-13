#pragma once
/* one entry of the enums table; 0x84 bytes: the enumerator name, then its value */
struct enumdef {
	char name[0x80];	/* +0 */
	int val;		/* +0x80 */
};
