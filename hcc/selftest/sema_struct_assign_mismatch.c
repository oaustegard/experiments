// EXPECT-ERROR: [parse]
struct A { int i; };
struct B { int i; };
int main(void) {
    struct A a;
    struct B b;
    a = b;
    return 0;
}
