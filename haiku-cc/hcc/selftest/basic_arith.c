// EXPECT-EXIT: 0
int check(long got, long want) { return got != want; }
int main(void) {
    int a = 7, b = 3;
    unsigned u = 0xffffffffu;
    long l = -5;
    int bad = 0;
    bad += check(a + b, 10); bad += check(a - b, 4); bad += check(a * b, 21);
    bad += check(a / b, 2); bad += check(a % b, 1); bad += check(-a / b, -2);
    bad += check(a & b, 3); bad += check(a | b, 7); bad += check(a ^ b, 4);
    bad += check(a << 4, 112); bad += check(-16 >> 2, -4); bad += check(u >> 28, 15);
    bad += check(u + 1, 0); bad += check(l * 3, -15); bad += check(~0, -1);
    bad += check(a < b, 0); bad += check(a >= b, 1); bad += check(a == 7 && b == 3, 1);
    bad += check(a != 7 || b != 3, 0); bad += check(!a, 0);
    bad += check((char)300, 44); bad += check((unsigned char)-1, 255);
    bad += check((short)70000, 4464); bad += check(-1 < 0u, 0);
    return bad;
}
