// EXPECT-EXIT: 0
// GNU #assert: #pred(answer) and #pred tests in #if; answers are not macro-expanded.
#define def 1
#assert abc (def)
#assert sp ( a  b )
#if !#abc(def) || #abc(other) || !#sp(a b) || !#sp
#error bad
#endif
#unassert abc
#if #abc || #abc(def)
#error bad2
#endif
int main(void) { return 0; }
