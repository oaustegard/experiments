// EXPECT-EXIT: 0
// Post-increment of a 1-bit bit-field: the old value must be returned,
// not recomputed as (new - 1) after the wrapping store.
struct s { unsigned a : 1; };
int main(void) {
    struct s x = { 1 };
    unsigned v = x.a++;
    if (v != 1 || x.a != 0) return 1;
    v = x.a--;
    if (v != 0 || x.a != 1) return 2;
    return 0;
}
