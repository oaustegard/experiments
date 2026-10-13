#pragma once
/* table of predefined macros: {name, body} pairs, NULL name ends */
struct def_macro {
	const char *name;
	const char *val;
};
extern struct def_macro def_macros[];
