#pragma once
#include "regex_t.h"
#include "mbox.h"

static char kwd[512];
static regex_t kwd_re;
static char kwd_hdr[512];
static struct mbox *mbox;
static int pos;
