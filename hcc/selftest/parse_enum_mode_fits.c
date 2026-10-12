// EXPECT-EXIT: 0
enum a { X = -1, Y = 127 } __attribute__((__mode__(__QI__)));
enum b { P = 255 } __attribute__((__mode__(__byte__)));
int main(void) { return (X + Y + P) - 381; }
