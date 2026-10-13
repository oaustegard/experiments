#pragma once
struct tr_cmd {
  char *name;
  void (*fn)(struct tr_args *);
  void (*mk)(struct tr_args *);
};
