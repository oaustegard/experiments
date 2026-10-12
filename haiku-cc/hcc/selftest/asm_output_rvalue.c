// EXPECT-ERROR: [parse]
struct x { int s; };
void f(struct x *a) {
  __asm__("" : "=r"((unsigned short)a->s));
}
