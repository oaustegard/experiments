#include "tr_cmd.h"

static void mkargs_ds();
static void mkargs_eol();
static void mkargs_null();
static void mkargs_ochar();
static void mkargs_reg1();
void tr_ab();
static void tr_ad();
static void tr_af();
static void tr_as();
static void tr_bd();
static void tr_blm();
void tr_bp();
void tr_br();
static void tr_c2();
static void tr_cc();
void tr_ce();
void tr_ch();
static void tr_char();
static void tr_chop();
void tr_cl();
static void tr_co();
static void tr_coa();
static void tr_coi();
static void tr_coo();
static void tr_cp();
static void tr_cs();
static void tr_de();
void tr_di();
void tr_divbeg();
void tr_divend();
void tr_divvs();
static void tr_ds();
void tr_dt();
static void tr_dv();
static void tr_ec();
static void tr_el();
void tr_em();
static void tr_eo();
static void tr_eos();
void tr_ev();
void tr_evc();
static void tr_ex();
void tr_fc();
static void tr_ff();
static void tr_ffsc();
void tr_fi();
static void tr_fmap();
void tr_fp();
void tr_fspecial();
void tr_ft();
static void tr_fzoom();
static void tr_hc();
void tr_hcode();
static void tr_hlm();
void tr_hpf();
void tr_hpfa();
void tr_hw();
static void tr_hy();
static void tr_hycost();
static void tr_hydash();
static void tr_hystop();
static void tr_if();
static void tr_ig();
void tr_in();
void tr_in2();
static void tr_it();
static void tr_kn();
void tr_l2r();
static void tr_lc();
static void tr_lf();
static void tr_lg();
void tr_ll();
static void tr_ls();
static void tr_lsm();
static void tr_lt();
static void tr_mc();
void tr_mk();
static void tr_na();
void tr_ne();
void tr_nf();
static void tr_nh();
static void tr_nm();
static void tr_nn();
static void tr_nr();
void tr_ns();
static void tr_nx();
static void tr_ochar();
void tr_os();
static void tr_pc();
static void tr_pl();
static void tr_pmll();
void tr_pn();
static void tr_po();
void tr_popren();
void tr_ps();
void tr_r2l();
static void tr_rchar();
static void tr_rm();
static void tr_rn();
static void tr_rr();
void tr_rs();
void tr_rt();
static void tr_shift();
static void tr_so();
void tr_sp();
static void tr_ss();
static void tr_ssh();
void tr_sv();
static void tr_sy();
void tr_ta();
static void tr_tc();
void tr_ti();
void tr_ti2();
static void tr_tkf();
static void tr_tl();
static void tr_tm();
static void tr_tr();
static void tr_vs();
void tr_wh();

static struct tr_cmd cmds[118] = {
  {"\007<", tr_divbeg, NULL},
  {"\007>", tr_divend, NULL},
  {"\007V", tr_divvs, NULL},
  {"\007P", tr_popren, NULL},
  {">>", tr_l2r, NULL},
  {"<<", tr_r2l, NULL},
  {"ab", tr_ab, mkargs_eol},
  {"ad", tr_ad, NULL},
  {"af", tr_af, NULL},
  {"am", tr_de, mkargs_reg1},
  {"as", tr_as, mkargs_ds},
  {"bd", tr_bd, NULL},
  {"blm", tr_blm, NULL},
  {"bp", tr_bp, NULL},
  {"br", tr_br, NULL},
  {"c2", tr_c2, NULL},
  {"cc", tr_cc, NULL},
  {"ce", tr_ce, NULL},
  {"ch", tr_ch, NULL},
  {"char", tr_char, mkargs_ds},
  {"chop", tr_chop, mkargs_reg1},
  {"cl", tr_cl, NULL},
  {"co", tr_co, NULL},
  {"co+", tr_coa, NULL},
  {"co<", tr_coi, mkargs_ds},
  {"co>", tr_coo, mkargs_ds},
  {"cp", tr_cp, NULL},
  {"cs", tr_cs, NULL},
  {"da", tr_di, NULL},
  {"de", tr_de, mkargs_reg1},
  {"di", tr_di, NULL},
  {"ds", tr_ds, mkargs_ds},
  {"dt", tr_dt, NULL},
  {"dv", tr_dv, mkargs_eol},
  {"ec", tr_ec, NULL},
  {"el", tr_el, mkargs_null},
  {"em", tr_em, NULL},
  {"eo", tr_eo, NULL},
  {"eos", tr_eos, NULL},
  {"ev", tr_ev, NULL},
  {"evc", tr_evc, NULL},
  {"ex", tr_ex, NULL},
  {"fc", tr_fc, NULL},
  {"ff", tr_ff, NULL},
  {"fi", tr_fi, NULL},
  {"fl", tr_br, NULL},
  {"fmap", tr_fmap, NULL},
  {"fp", tr_fp, NULL},
  {"ffsc", tr_ffsc, NULL},
  {"fspecial", tr_fspecial, NULL},
  {"ft", tr_ft, NULL},
  {"fzoom", tr_fzoom, NULL},
  {"hc", tr_hc, NULL},
  {"hcode", tr_hcode, NULL},
  {"hlm", tr_hlm, NULL},
  {"hpf", tr_hpf, NULL},
  {"hpfa", tr_hpfa, NULL},
  {"hy", tr_hy, NULL},
  {"hycost", tr_hycost, NULL},
  {"hydash", tr_hydash, NULL},
  {"hystop", tr_hystop, NULL},
  {"hw", tr_hw, NULL},
  {"ie", tr_if, mkargs_null},
  {"if", tr_if, mkargs_null},
  {"ig", tr_ig, NULL},
  {"in", tr_in, NULL},
  {"in2", tr_in2, NULL},
  {"it", tr_it, NULL},
  {"kn", tr_kn, NULL},
  {"lc", tr_lc, NULL},
  {"lf", tr_lf, NULL},
  {"lg", tr_lg, NULL},
  {"ll", tr_ll, NULL},
  {"ls", tr_ls, NULL},
  {"lsm", tr_lsm, NULL},
  {"lt", tr_lt, NULL},
  {"mc", tr_mc, NULL},
  {"mk", tr_mk, NULL},
  {"na", tr_na, NULL},
  {"ne", tr_ne, NULL},
  {"nf", tr_nf, NULL},
  {"nh", tr_nh, NULL},
  {"nm", tr_nm, NULL},
  {"nn", tr_nn, NULL},
  {"nr", tr_nr, mkargs_reg1},
  {"ns", tr_ns, NULL},
  {"nx", tr_nx, NULL},
  {"ochar", tr_ochar, mkargs_ochar},
  {"os", tr_os, NULL},
  {"pc", tr_pc, NULL},
  {"pl", tr_pl, NULL},
  {"pmll", tr_pmll, NULL},
  {"pn", tr_pn, NULL},
  {"po", tr_po, NULL},
  {"ps", tr_ps, NULL},
  {"rchar", tr_rchar, NULL},
  {"rm", tr_rm, NULL},
  {"rn", tr_rn, NULL},
  {"rr", tr_rr, NULL},
  {"rs", tr_rs, NULL},
  {"rt", tr_rt, NULL},
  {"shift", tr_shift, NULL},
  {"so", tr_so, NULL},
  {"sp", tr_sp, NULL},
  {"ss", tr_ss, NULL},
  {"ssh", tr_ssh, NULL},
  {"sv", tr_sv, NULL},
  {"sy", tr_sy, mkargs_eol},
  {"ta", tr_ta, NULL},
  {"tc", tr_tc, NULL},
  {"ti", tr_ti, NULL},
  {"ti2", tr_ti2, NULL},
  {"tkf", tr_tkf, NULL},
  {"tl", tr_tl, mkargs_null},
  {"tm", tr_tm, mkargs_eol},
  {"tr", tr_tr, mkargs_eol},
  {"vs", tr_vs, NULL},
  {"wh", tr_wh, NULL},
};
