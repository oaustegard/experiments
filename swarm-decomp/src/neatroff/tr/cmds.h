#pragma once

#include "ext.h"

/* the command table (cmds.c) */
static struct tr_cmd cmds[118];
#pragma once

#include "ext.h"

/* functions of this TU that the command table refers to */
static void mkargs_ds();
static void mkargs_eol();
static void mkargs_null();
static void mkargs_ochar();
static void mkargs_reg1();
static void tr_ad();
static void tr_af();
static void tr_as();
static void tr_bd();
static void tr_blm();
static void tr_c2();
static void tr_cc();
static void tr_char();
static void tr_chop();
static void tr_co();
static void tr_coa();
static void tr_coi();
static void tr_coo();
static void tr_cp();
static void tr_cs();
static void tr_de();
static void tr_ds();
static void tr_dv();
static void tr_ec();
static void tr_el();
static void tr_eo();
static void tr_eos();
static void tr_ex();
static void tr_ff();
static void tr_ffsc();
static void tr_fmap();
static void tr_fzoom();
static void tr_hc();
static void tr_hlm();
static void tr_hy();
static void tr_hycost();
static void tr_hydash();
static void tr_hystop();
static void tr_if();
static void tr_ig();
static void tr_it();
static void tr_kn();
static void tr_lc();
static void tr_lf();
static void tr_lg();
static void tr_ls();
static void tr_lsm();
static void tr_lt();
static void tr_mc();
static void tr_na();
static void tr_nh();
static void tr_nm();
static void tr_nn();
static void tr_nr();
static void tr_nx();
static void tr_ochar();
static void tr_pc();
static void tr_pl();
static void tr_pmll();
static void tr_po();
static void tr_rchar();
static void tr_rm();
static void tr_rn();
static void tr_rr();
static void tr_shift();
static void tr_so();
static void tr_ss();
static void tr_ssh();
static void tr_sy();
static void tr_tc();
static void tr_tkf();
static void tr_tl();
static void tr_tm();
static void tr_tr();
static void tr_vs();
