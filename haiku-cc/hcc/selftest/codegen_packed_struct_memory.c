// EXPECT-EXIT: 0
// Packed struct with a misaligned member is MEMORY class: returned via sret
// and passed by value without register splitting.
extern void abort(void);
struct __attribute__((packed)) S { unsigned int l; unsigned short i : 4, j : 1, k : 11; };
struct __attribute__((packed)) X { unsigned int k : 12, j : 13, i : 7; long double l; };
struct __attribute__((packed)) P { char c; long long v; };
struct P mk(long long v) { struct P p; p.c = 1; p.v = v; return p; }
long long get(struct P p) { return p.v + p.c; }
struct S ret(struct S s) { return s; }
int main(void) {
    struct S s;
    s.l = 7; s.i = 3; s.j = 1; s.k = 100;
    struct S t = ret(s);
    if (t.l != 7 || t.i != 3 || t.j != 1 || t.k != 100) abort();
    if (get(mk(40)) != 41) abort();
    return 0;
}
