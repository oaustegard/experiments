#pragma once
/* one entry of the cpp macro table; 0x188c bytes */
struct macro {
	char name[128];		/* +0 */
	char body[2048];	/* +128 */
	char arg[32][128];	/* +2176 */
	int nargs;		/* +6272 */
	int isfunc;		/* +6276 */
	int flags;		/* +6280 */
};
