static const char *match (MatchState *ms, const char *s, const char *p) {
  if (__builtin_expect(((ms->matchdepth-- == 0) != 0), 0))
    luaL_error(ms->L, "pattern too complex");
  init:
  if (p != ms->p_end) {
    switch (*p) {
      case '(': {
        if (*(p + 1) == ')')
          s = start_capture(ms, s, p + 2, CAP_POSITION);
        else
          s = start_capture(ms, s, p + 1, CAP_UNFINISHED);
        break;
      }
      case ')': {
        s = end_capture(ms, s, p + 1);
        break;
      }
      case '$': {
        if ((p + 1) != ms->p_end)
          goto dflt;
        s = (s == ms->src_end) ? s : NULL;
        break;
      }
      case L_ESC: {
        switch (*(p + 1)) {
          case 'b': {
            s = matchbalance(ms, s, p + 2);
            if (s != NULL) {
              p += 4; goto init;
            }
            break;
          }
          case 'f': {
            const char *ep; char previous;
            p += 2;
            if (__builtin_expect(((*p != '[') != 0), 0))
              luaL_error(ms->L, "missing '[' after '%%f' in pattern");
            ep = classend(ms, p);
            previous = (s == ms->src_init) ? '\0' : *(s - 1);
            if (!matchbracketclass((unsigned char)previous, p, ep - 1) &&
               matchbracketclass((unsigned char)*s, p, ep - 1)) {
              p = ep; goto init;
            }
            s = NULL;
            break;
          }
          case '0': case '1': case '2': case '3':
          case '4': case '5': case '6': case '7':
          case '8': case '9': {
            s = match_capture(ms, s, (unsigned char)*(p + 1));
            if (s != NULL) {
              p += 2; goto init;
            }
            break;
          }
          default: goto dflt;
        }
        break;
      }
      default: dflt: {
        const char *ep = classend(ms, p);
        if (!singlematch(ms, s, p, ep)) {
          if (*ep == '*' || *ep == '?' || *ep == '-') {
            p = ep + 1; goto init;
          }
          else
            s = NULL;
        }
        else {
          switch (*ep) {
            case '?': {
              const char *res;
              if ((res = match(ms, s + 1, ep + 1)) != NULL)
                s = res;
              else {
                p = ep + 1; goto init;
              }
              break;
            }
            case '+':
              s++;
              /* FALLTHROUGH */
            case '*':
              s = max_expand(ms, s, p, ep);
              break;
            case '-':
              s = min_expand(ms, s, p, ep);
              break;
            default:
              s++; p = ep; goto init;
          }
        }
        break;
      }
    }
  }
  ms->matchdepth++;
  return s;
}
