#include <signal.h>
#include <stdio.h>
#include <sys/stat.h>
#include "fn_ex.h"
#include "fn_me.h"
#include "fn_mk.h"
#include "fn_pg.h"
#include "fn_pn.h"
int main(int argc, char **argv)
{
	struct stat st;
	char *cmd;

	signal(SIGPIPE, SIG_IGN);
	if (argv[1] == NULL)
		puts("usage: neatmail command [options]\n\ncommands:\n   ex  \texecute commands on an mbox\n   mk  \tlist the messages in an mbox\n   pg  \tpage a message of an mbox\n   me  \tencode mime message headers\n   pn  \tprune an mbox\n   ns  \tcheck mboxes for new mails");
	cmd = argv[1];
	if (cmd != NULL) {
		if (strcmp(cmd, "mk") == 0)
			return mk(argv + 2);
		if (strcmp("ex", cmd) == 0)
			return ex(argv + 2);
		if (strcmp("pg", cmd) == 0)
			return pg(argv + 2);
		if (strcmp("me", cmd) == 0)
			return me(argv + 2);
		if (strcmp("pn", cmd) == 0)
			return pn(argv + 2);
		if (strcmp("ns", cmd) == 0) {
			for (argv += 2; *argv; argv++)
				if (stat(*argv, &st) != -1 && st.st_mtime > st.st_atime)
					puts(*argv);
		}
	}
	return 0;
}
