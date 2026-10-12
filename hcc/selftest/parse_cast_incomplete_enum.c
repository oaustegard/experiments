// EXPECT-ERROR: conversion to incomplete type
enum T;
int main(void) {
    return sizeof((enum T)1);
}
