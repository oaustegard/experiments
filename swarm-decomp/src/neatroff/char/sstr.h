#pragma once

/* sstr stack state, shared by the sstr_* and charread/charnext functions */
static char *sstr_s;
static int sstr_n;
static char *sstr_bufs[32];
