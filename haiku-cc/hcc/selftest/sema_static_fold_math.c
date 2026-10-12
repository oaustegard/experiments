// EXPECT-EXIT: 0
// Static initializers: libm calls with constant args fold (GNU), and float
// constants stored into integer statics are converted.
double atan(double);
double nan(const char *);
const double pi = 4 * atan(1.0);
const double nn = nan("");
int d[1] = {0.5};
int x = 0.5 * 4;
int main(void) {
    if (pi < 3.14 || pi > 3.15) return 1;
    if (nn == nn) return 2;
    if (d[0] != 0 || x != 2) return 3;
    return 0;
}
