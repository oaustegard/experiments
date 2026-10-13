#pragma once

/* integer set insert; declared here, not in types/, because other neatroff
   TUs declare iset_put with a struct iset pointer */
void iset_put(void *set, int i, int v);
