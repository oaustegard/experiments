// EXPECT-EXIT: 0
// _FloatN/_FloatNx type names map to float types; C23 static_assert keyword
// (message optional).
_Static_assert(sizeof(_Float32) == 4);
_Static_assert(sizeof(_Float64) == 8);
int main(void) {
    _Float32 f = 1.5f;
    _Float64 d = f;
    _Float64x x = d;
    _Static_assert(1, "ok");
    return (int)(d + x) - 3;
}
