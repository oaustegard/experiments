#!/bin/sh
set -e
cd "$(dirname "$0")"
mojo build mdifflib/kernel.mojo --emit shared-lib -o mdifflib/_kernel.so
