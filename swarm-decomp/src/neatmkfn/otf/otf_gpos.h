#pragma once

static void otf_gpos(const unsigned char *t, int unused);

/* defined later in this TU; this file sorts before them, so declare them here */
static int valuerecord_small(int mask, const unsigned char *p);
static void valuerecord_print(int mask, const unsigned char *p);
static int classdef(const unsigned char *t, int *gl, int *cls);
static int ggrp_class(int *gl, int *cls, int n, int k);
static char glyph_name[65536][64];
static int glyph_wid[65536];
