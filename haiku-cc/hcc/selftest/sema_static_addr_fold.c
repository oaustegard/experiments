// EXPECT-EXIT: 0
// Static initializers folding address comparisons, same-object address
// differences and scalar file-scope compound literals.
extern void abort(void);
struct s { char p[2]; };
static struct s v;
int g = 5;
const int o1 = (int) ((void *) &v.p[1] - (void *) &v) + 1;
int n = (&g >= 0);
int z = (&g == 0);
int c = (int){7};
int main(void) {
    if (o1 != 2 || n != 1 || z != 0 || c != 7) abort();
    return 0;
}
