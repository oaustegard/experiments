// EXPECT-EXIT: 0
// EXPECT-FLAGS: -fplan9-extensions
// -fplan9-extensions: `struct S;` is an anonymous member; a pointer to S converts
// implicitly to a pointer to that member (adjusted by its offset); a typedef-named
// anonymous member is reachable by its typedef name.
struct A { char a; };
struct B { char b; struct A; char c; };
typedef struct { char d; } D;
struct E { char b; union { D; }; char e; };

static void set_a(struct A *p) { p->a = 7; }
static void set_d(D *p) { p->d = 4; }

int main(void) {
    struct B b;
    struct A *pa;
    struct E e;
    D *pd;
    b.b = 1; b.c = 2;
    set_a(&b);
    if (b.a != 7 || b.b != 1 || b.c != 2) return 1;
    pa = &b;
    if (pa->a != 7) return 2;
    e.b = 1; e.e = 3;
    set_d(&e);
    if (e.d != 4 || e.D.d != 4 || e.b != 1 || e.e != 3) return 3;
    pd = &e;
    if (pd->d != 4) return 4;
    return 0;
}
