#pragma once

/* one alternate-name entry: utf-8 character, ascii alias, name */
struct alt_ent {
	char *utf8;
	char *ascii;
	char *name;
	char *pad[5];
};
