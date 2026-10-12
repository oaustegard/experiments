// EXPECT-EXIT: 0
// mode(word)/mode(QI) on typedefs, enums (incl. attribute after the enum body)
// and declarations resize the type; aligned on a variable is seen by __alignof__.
typedef int tword __attribute__((mode(word)));
typedef enum { C1 = 1 } C __attribute__((mode(QI)));
typedef enum { D1 = 1 } __attribute__((mode(word))) D;
char x __attribute__((aligned(128)));
int s1[sizeof(tword) == 8 ? 1 : -1];
int s2[sizeof(C) == 1 ? 1 : -1];
int s3[sizeof(D) == 8 ? 1 : -1];
int a1[__alignof__(x) == 128 ? 1 : -1];
int main(void) { return 0; }
