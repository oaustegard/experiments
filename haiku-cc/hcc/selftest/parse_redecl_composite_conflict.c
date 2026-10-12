// EXPECT-ERROR: [parse]
typedef int IA[];
typedef int IA5[5];
typedef int IA10[10];
typedef IA5 *IA5P;
typedef IA10 *IA10P;
void f(void) { extern IA5P a[]; }
IA5P a[] = { 0 };
extern IA10P a[];
int main(void) { return 0; }
