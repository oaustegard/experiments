// EXPECT-EXIT: 7
struct later;
typedef const struct later *cl;
struct later { int a; int b; };
int get(cl p) { return p->b; }
int main(void) {
    struct later s = { 1, 7 };
    return get(&s);
}
