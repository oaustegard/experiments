// EXPECT-EXIT: 0
typedef void va_type(const char *, ...);
typedef void printf_like(const char *, ...) [[gnu::format(printf, 1, 2)]];
typedef __typeof__(void (int) [[gnu::deprecated]]) fn_t;
fn_t *p;
int main(void) {
    return p == 0 ? 0 : 0;
}
