static void readpre(void)
{
	char *t;
	struct type x;
	long num;
	int ty;
	int np;
	long mark;
	int k;

	t = tok_see();
	if (*t == '&' && !t[1]) {
		tok_get();
		readpre();
		nts--;
		x = ts[nts];
		if (!x.tag)
			err("cannot use the address\n");
		x.ptr++;
		x.tag = 0;
		ts[nts] = x;
		nts++;
	} else {
		t = tok_see();
		if (*t == '*' && !t[1]) {
			tok_get();
			readpre();
			nts--;
			x = ts[nts];
			if ((x.flags & 1) && x.ptr == 0) {
				x = ((struct array *)arrays)[x.arr].t;
				x.ptr++;
			}
			if (x.ptr == 0) {
				err("dereferencing non-pointer\n");
				if (x.tag)
					o_deref(x.kind);
			} else if (x.tag) {
				o_deref(8);
			}
			x.ptr--;
			x.tag = 1;
			ts[nts] = x;
			nts++;
		} else {
			t = tok_see();
			if (*t == '!' && !t[1]) {
				tok_get();
				readpre();
				ts_de(1);
				nts--;
				o_uop(0x102);
				ts[nts].kind = 0x14;
				ts[nts].flags = 0;
				ts[nts].ptr = 0;
				ts[nts].tag = 0;
				nts++;
			} else {
				t = tok_see();
				if (*t == '+' && !t[1]) {
					tok_get();
					readpre();
					ts_de(1);
					x = ts[nts - 1];
					if (x.ptr)
						k = 8;
					else {
						k = (x.kind & 0xf) < 4 ? 4 : (x.kind & 0xf);
						if (x.kind & 0x10)
							k |= 0x10;
						else if (!(x.kind & 0xc))
							k = 4 | 0x10;
					}
					ts[nts - 1].kind = k;
					ts[nts - 1].flags = 0;
					ts[nts - 1].ptr = 0;
					ts[nts - 1].tag = 0;
				} else {
					t = tok_see();
					if (*t == '-' && !t[1]) {
						tok_get();
						readpre();
						ts_de(1);
						nts--;
						x = ts[nts];
						o_uop(0x100);
						if (x.ptr)
							k = 8;
						else {
							k = (x.kind & 0xf) < 4 ? 4 : (x.kind & 0xf);
							if (x.kind & 0x10)
								k |= 0x10;
							else if (!(x.kind & 0xc))
								k = 4 | 0x10;
						}
						ts[nts].kind = k;
						ts[nts].flags = 0;
						ts[nts].ptr = 0;
						ts[nts].tag = 0;
						nts++;
					} else {
						t = tok_see();
						if (*t == '~' && !t[1]) {
							tok_get();
							readpre();
							ts_de(1);
							nts--;
							x = ts[nts];
							o_uop(0x101);
							if (x.ptr)
								k = 8;
							else {
								k = (x.kind & 0xf) < 4 ? 4 : (x.kind & 0xf);
								if (x.kind & 0x10)
									k |= 0x10;
								else if (!(x.kind & 0xc))
									k = 4 | 0x10;
							}
							ts[nts].kind = k;
							ts[nts].flags = 0;
							ts[nts].ptr = 0;
							ts[nts].tag = 0;
							nts++;
						} else {
							t = tok_see();
							if (*t == '+' && t[1] == '+' && !t[2]) {
								tok_get();
								inc_pre(0x10);
							} else {
								t = tok_see();
								if (*t == '-' && t[1] == '-' && !t[2]) {
									tok_get();
									inc_pre(0x11);
								} else {
									t = tok_see();
									if (strcmp("sizeof", t) == 0) {
										tok_get();
										t = tok_see();
										if (*t == '(' && !t[1]) {
											tok_get();
											if (readname(&x, 0, 0) == 0) {
												o_num(type_totsz(&x));
												ts[nts].kind = 8;
												ts[nts].flags = 0;
												ts[nts].ptr = 0;
												ts[nts].tag = 0;
												nts++;
											} else {
												mark = o_mark();
												readexpr();
												o_back(mark);
												nts--;
												x = ts[nts];
												o_tmpdrop(1);
												o_num(type_totsz(&x));
												ts[nts].kind = 8;
												ts[nts].flags = 0;
												ts[nts].ptr = 0;
												ts[nts].tag = 0;
												nts++;
											}
											t = tok_get();
											if (*t != ')' || t[1])
												err("syntax error (expected <%s> but got <%s>)\n", ")", t);
										} else if (readname(&x, 0, 0) == 0) {
											o_num(type_totsz(&x));
											ts[nts].kind = 8;
											ts[nts].flags = 0;
											ts[nts].ptr = 0;
											ts[nts].tag = 0;
											nts++;
										} else {
											mark = o_mark();
											readpre();
											o_back(mark);
											nts--;
											x = ts[nts];
											o_tmpdrop(1);
											o_num(type_totsz(&x));
											ts[nts].kind = 8;
											ts[nts].flags = 0;
											ts[nts].ptr = 0;
											ts[nts].tag = 0;
											nts++;
										}
									} else {
if (tok_grp() == '0') {
										ty = tok_num(tok_get(), &num);
										o_num(num);
										ts[nts].kind = ty;
										ts[nts].flags = 0;
										ts[nts].ptr = 0;
										ts[nts].tag = 0;
										nts++;
									} else if (tok_grp() == '"') {
										char *s;
										struct type str;
										struct type elt;
										int len;
										int a;

										memset(&str, 0, sizeof str);
										memset(&elt, 0, sizeof elt);
										elt.kind = 0x11;
										s = tok_get();
										len = tok_len();
										a = array_add(&elt, len - 1);
										o_sym(tmp_str(s + 1, len - 2));
										str.arr = a;
										str.flags = 1;
										ts[nts++] = str;
									} else if (tok_grp() == 'a') {
										struct decl d;
										int i;

										memset(&d, 0, sizeof d);
										snprintf(tok_previden, 0x80, "%s", tok_get());
										strcpy(d.name, tok_previden);
										if (ncexpr || caseexpr || strcmp(":", tok_see()) != 0) {
											for (i = locals_n - 1; i >= 0; i--) {
												struct decl *l = (struct decl *)locals + i;

												if (strcmp(l->name, d.name) == 0) {
													o_local(*(long *)l->loc);
													ts[nts] = l->t;
													ts[nts].tag = 1;
													nts++;
													goto postfix;
												}
											}
											for (i = globals_n - 1; i >= 0; i--) {
												struct decl *g = (struct decl *)globals + i;

												if (strcmp(d.name, g->name) == 0) {
													o_sym(g->pre[0] ? g->pre : g->name);
													ts[nts] = g->t;
													ts[nts].tag = 1;
													nts++;
													goto postfix;
												}
											}
											for (i = enums_n - 1; i >= 0; i--) {
												struct enumdef *e = (struct enumdef *)enums + i;

												if (strcmp(d.name, e->name) == 0) {
													o_num(e->val);
													ts[nts].kind = 0x14;
													ts[nts].flags = 0;
													ts[nts].ptr = 0;
													ts[nts].tag = 0;
													nts++;
													goto postfix;
												}
											}
											if (strcmp("(", tok_see()) != 0)
												err("unknown symbol <%s>\n", d.name);
											global_add(&d);
											o_sym(d.name);
											ts[nts].kind = 8;
											ts[nts].flags = 0;
											ts[nts].ptr = 0;
											ts[nts].tag = 0;
											nts++;
										}
									} else {
										t = tok_see();
										if (*t == '(' && !t[1]) {
											tok_get();
											if (readname(&x, 0, 0) != 0) {
												readexpr();
												for (;;) {
													t = tok_see();
													if (*t == ')' && !t[1]) {
														tok_get();
														break;
													}
													t = tok_get();
													if (*t != ',' || t[1])
														err("syntax error (expected <%s> but got <%s>)\n", ",", t);
													nts--;
													o_tmpdrop(1);
													readexpr();
												}
											} else {
												t = tok_get();
												if (strcmp(")", t) != 0)
													err("syntax error (expected <%s> but got <%s>)\n", ")", t);
												readpre();
												ts_de(1);
												k = ts[nts - 1].ptr;
												ts[nts - 1] = x;
												if (x.ptr == 0)
													o_cast(x.kind);
												else if (k == 0)
													o_cast(8);
											}
										}
									}
									postfix:
									for (;;) {
										t = tok_see();
										if (*t == '[' && !t[1]) {
											long n;

											tok_get();
											readexpr();
											t = tok_get();
											if (*t != ']' || t[1])
												err("syntax error (expected <%s> but got <%s>)\n", "]", t);
											ts_de(1);
											nts -= 2;
											x = ts[nts];
											if ((x.flags & 1) && x.ptr == 0) {
												x = ((struct array *)arrays)[x.arr].t;
											} else {
												if (x.tag) {
													o_tmpswap();
													o_deref(x.ptr ? 8 : x.kind);
													o_tmpswap();
												}
												x.ptr--;
											}
											n = type_totsz(&x);
											x.tag = 1;
											if (n > 1) {
												o_num(n);
												o_bop(0x40);
											}
											o_bop(0x10);
											ts[nts] = x;
											nts++;
											continue;
										}
										t = tok_see();
										if (*t == '(' && !t[1]) {
											int nargs = 0;
											int fl;
											int arr;
											struct func *f;

											tok_get();
											nts--;
											fl = ts[nts].flags & 4;
											k = ts[nts].ptr;
											arr = ts[nts].arr;
											if (k > 0 && fl)
												o_deref(8);
											t = tok_see();
											if (*t == ')' && !t[1])
												nargs = 0;
											else {
												for (;;) {
													readexpr();
													nargs++;
													ts_de(1);
													nts--;
													t = tok_see();
													if (*t != ',' || t[1])
														break;
													tok_get();
												}
											}
											t = tok_get();
											if (*t != ')' || t[1])
												err("syntax error (expected <%s> but got <%s>)\n", ")", t);
											if (fl) {
												f = (struct func *)funcs + arr;
												if (f) {
													o_call(nargs, f->param[32].ptr ? 8 : f->param[32].kind);
													if (f->param[32].ptr)
														o_cast(8);
													else if (f->param[32].kind)
														o_cast(f->param[32].kind);
													ts[nts] = f->param[32];
													nts++;
													continue;
												}
											}
											o_call(nargs, 0x14);
											ts[nts].kind = 0x14;
											ts[nts].flags = 0;
											ts[nts].ptr = 0;
											ts[nts].tag = 0;
											nts++;
											continue;
										}
										t = tok_see();
										if (*t == '+' && t[1] == '+' && !t[2]) {
											tok_get();
											inc_post(0x10);
											continue;
										}
										t = tok_see();
										if (*t == '-' && t[1] == '-' && !t[2]) {
											tok_get();
											inc_post(0x11);
											continue;
										}
										t = tok_see();
										if (*t == '.' && !t[1]) {
											tok_get();
											readfield();
											continue;
										}
										t = tok_see();
										if (*t == '-' && t[1] == '>' && !t[2]) {
											tok_get();
											ts_de(1);
											readfield();
											continue;
										}
										break;
									}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
