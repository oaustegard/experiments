// EXPECT-EXIT: 0
// #pragma scalar_storage_order big-endian applies to later struct definitions only.
struct S1 { int i; };
#pragma scalar_storage_order big-endian
struct S2 { int i; };
#pragma scalar_storage_order default
struct S3 { int i; };
struct S1 a = { 0x12345678 };
struct S2 b = { 0x12345678 };
struct S3 c = { 0x12345678 };
unsigned char le[4] = { 0x78, 0x56, 0x34, 0x12 };
unsigned char be[4] = { 0x12, 0x34, 0x56, 0x78 };
int main(void) {
    if (__builtin_memcmp(&a, le, 4)) return 1;
    if (__builtin_memcmp(&b, be, 4)) return 2;
    if (__builtin_memcmp(&c, le, 4)) return 3;
    return 0;
}
