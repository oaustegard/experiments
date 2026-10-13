#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "regex_t.h"

/* regmatch entry: 16 bytes, two 64-bit offsets */
struct rmatch {
	long rm_so;
	long rm_eo;
};

/* backtrack stack entry: 12 bytes */
struct rbt {
	int pos;	/* offset of the text position from the string start */
	int nlog;	/* capture undo-log depth to restore */
	int state;	/* program slot to resume at */
};

/* capture undo-log entry: 8 bytes */
struct rundo {
	int slot;	/* 2 * group (+1 for the end mark) */
	int pos;	/* offset from the string start */
};

int regexec(const regex_t *re, const char *s, int nmatch, struct rmatch *pmatch, int eflags)
{
	struct rprog *prog = re->prog;
	struct rslot *slots = prog->slots;
	struct rbt stk0[128];
	struct rundo log0[128];
	struct rbt *stk = stk0;
	struct rundo *log = log0;
	const char *start = s;
	const char *cur, *p, *prev, *nxt, *q;
	const char *str;
	struct rslot *sl;
	int nstk = 0, capstk = 128;
	int nlog = 0, caplog = 128;
	int nsub, flags, nb, state, ret, i, c, pc, tc, off;
	unsigned char ch;

	nsub = (eflags & 2) ? 0 : nmatch;
	flags = prog->cflags | eflags;
	nb = eflags & 0x40;

	for (i = 0; i < nmatch; i++) {
		pmatch[i].rm_so = -1;
		pmatch[i].rm_eo = -1;
	}
	if (!*s)
		return 1;

	prev = p = s;
loop:
	if (nb && prev != s && *prev == '\n') {
		ret = 1;
		goto out;
	}
	cur = p;
	nxt = p + uc_len(p);
	nlog = 0;
	nstk = 0;
	state = 0;

step:
	sl = &slots[state];
	switch (sl->flag) {
	case 'm':
		if (sl->c18 < 2 * nsub) {
			if (nlog >= caplog) {
				struct rundo *g = malloc(sizeof(struct rundo) * 2 * caplog);
				if (g) {
					memcpy(g, log, sizeof(struct rundo) * caplog);
					if (log != log0)
						free(log);
					log = g;
					caplog *= 2;
				}
			}
			if (nlog < caplog) {
				log[nlog].slot = sl->c18;
				log[nlog].pos = (int)(cur - start);
				nlog++;
			}
		}
		state++;
		goto step;
	case 'j':
		state = sl->link;
		goto step;
	case 'f':
		if (nstk >= capstk) {
			struct rbt *g = malloc(sizeof(struct rbt) * 2 * capstk);
			if (!g)
				goto fail;
			memcpy(g, stk, sizeof(struct rbt) * capstk);
			if (stk != stk0)
				free(stk);
			stk = g;
			capstk *= 2;
		}
		stk[nstk].pos = (int)(cur - start);
		stk[nstk].nlog = nlog;
		stk[nstk].state = sl->link;
		nstk++;
		state++;
		goto step;
	case 'q':
		for (i = 0; i < nlog; i++) {
			struct rmatch *m = &pmatch[log[i].slot >> 1];
			if (log[i].slot & 1)
				m->rm_eo = log[i].pos;
			else
				m->rm_so = log[i].pos;
		}
		ret = 0;
		goto out;
	case 0:
		break;
	default:
		goto fail;
	}

	str = sl->str;
	switch (sl->c00) {
	case 0:
		if (!(flags & 4)) {
			const char *t = cur, *u = str;
			while (*u) {
				if (*t != *u)
					goto fail;
				t++;
				u++;
			}
			cur = t;
			state++;
			goto step;
		}
		if (!*str) {
			state++;
			goto step;
		}
		for (off = 0; ; ) {
			pc = uc_dec(str + off);
			tc = uc_dec(cur + off);
			if (pc <= 0x7f && isupper(pc))
				pc = tolower(pc);
			if (tc <= 0x7f && isupper(tc))
				tc = tolower(tc);
			if (pc != tc)
				goto fail;
			off += uc_len(str + off);
			if (!str[off])
				break;
		}
		cur += off;
		state++;
		goto step;
	case '.':
		ch = (unsigned char)*cur;
		if (!ch)
			goto fail;
		if (ch == '\n' && (flags & 8))
			goto fail;
		cur += uc_len(cur);
		state++;
		goto step;
	case '[':
		c = uc_dec(cur);
		if (!c)
			goto fail;
		if (c == '\n' && (flags & 8) && str[1] == '^')
			goto fail;
		cur += uc_len(cur);
		if (brk_match(str + 1, c, flags) == 0) {
			state++;
			goto step;
		}
		goto fail;
	case '^':
		if (cur == start) {
			if (flags & 0x10)
				goto fail;
		} else {
			if (cur <= start || cur[-1] != '\n' || !(flags & 8))
				goto fail;
		}
		state++;
		goto step;
	case '$':
		ch = (unsigned char)*cur;
		if (ch == 0) {
			if (flags & 0x20)
				goto fail;
		} else if (ch == '\n') {
			if (!(flags & 8))
				goto fail;
		} else {
			goto fail;
		}
		state++;
		goto step;
	case '<':
		if (cur != start) {
			q = cur - 1;
			while (q > start && (*q & 0xc0) == 0x80)
				q--;
			ch = (unsigned char)*q;
			if (isalnum(ch) || ch == '_' || ch > 0x7f)
				goto fail;
		}
		ch = (unsigned char)*cur;
		if (!(isalnum(ch) || ch == '_' || ch > 0x7f))
			goto fail;
		state++;
		goto step;
	case '>':
		if (cur == start)
			goto fail;
		q = cur - 1;
		while (q > start && (*q & 0xc0) == 0x80)
			q--;
		ch = (unsigned char)*q;
		if (!(isalnum(ch) || ch == '_' || ch > 0x7f))
			goto fail;
		ch = (unsigned char)*cur;
		if (ch != 0 && (isalnum(ch) || ch == '_' || ch > 0x7f))
			goto fail;
		state++;
		goto step;
	default:
		goto fail;
	}

fail:
	if (nstk == 0)
		goto next_attempt;
	nstk--;
	cur = start + stk[nstk].pos;
	nlog = stk[nstk].nlog;
	state = stk[nstk].state;
	goto step;

next_attempt:
	prev = p;
	p = nxt;
	if (!*prev) {
		ret = 1;
		goto out;
	}
	goto loop;

out:
	if (log != log0)
		free(log);
	if (stk != stk0)
		free(stk);
	return ret;
}
