// EXPECT-EXIT: 6
[[gnu::noipa]] static int f(int x)
{
  [[maybe_unused]] int y = x + 1;
  switch (y) {
  case 3:
    [[fallthrough]];
  case 4:
    return y + 3;
  default:
    ;
  }
  return 0;
}

[[gnu::unused]] int g = 4;

int main(void)
{
  [[maybe_unused]] int r = f(2);
  return r;
}
