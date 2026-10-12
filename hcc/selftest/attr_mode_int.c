// EXPECT-EXIT: 0
// __attribute__((mode(QI/HI/SI/DI))) re-sizes an integer typedef; packed
// structs built from them must use the narrow sizes.
typedef unsigned int __attribute__((mode(QI))) u1_t;
typedef unsigned int __attribute__((mode(HI))) u2_t;
typedef int __attribute__((mode(SI))) s4_t;
typedef unsigned int __attribute__((mode(DI))) u8_t;
struct __attribute__((packed)) S { u1_t o[3]; u2_t x; };
int main(void) {
  if (sizeof(u1_t) != 1 || sizeof(u2_t) != 2 || sizeof(s4_t) != 4 || sizeof(u8_t) != 8)
    return 1;
  if (sizeof(struct S) != 5)
    return 2;
  if ((s4_t)-1 >= 0)
    return 3;
  return 0;
}
