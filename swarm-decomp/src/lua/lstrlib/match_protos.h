#pragma once
#include "MatchState.h"

#define L_ESC '%'
#define CAP_UNFINISHED (-1)
#define CAP_POSITION (-2)

static const char *classend (MatchState *ms, const char *p);
static int match_class (int c, int cl);
static int matchbracketclass (int c, const char *p, const char *ec);
static int singlematch (MatchState *ms, const char *s, const char *p, const char *ep);
static const char *matchbalance (MatchState *ms, const char *s, const char *p);
static const char *max_expand (MatchState *ms, const char *s, const char *p, const char *ep);
static const char *min_expand (MatchState *ms, const char *s, const char *p, const char *ep);
static const char *start_capture (MatchState *ms, const char *s, const char *p, int what);
static const char *end_capture (MatchState *ms, const char *s, const char *p);
static const char *match_capture (MatchState *ms, const char *s, int l);
static int check_capture (MatchState *ms, int l);
static int capture_to_close (MatchState *ms);
