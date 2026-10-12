// EXPECT-EXIT: 0
// ms_struct bit-fields: a bit-field of a different-size type starts after the
// previous storage unit; a zero-width bit-field after a plain member is ignored.
#define MS __attribute__((__ms_struct__))
struct A { unsigned short x:12; int y:1; unsigned short z:6; } MS;   /* 12 */
struct B { unsigned int a:26; unsigned char b:2; } MS;                /* 8 */
struct C { char a; int :0; char b; char c; } MS;                      /* 3 */
struct D { char a:8; int :0; char b; char c; } MS;                    /* 8 */
int main(void) {
    if (sizeof(struct A) != 12) return 1;
    if (sizeof(struct B) != 8) return 2;
    if (sizeof(struct C) != 3) return 3;
    if (sizeof(struct D) != 8) return 4;
    return 0;
}
