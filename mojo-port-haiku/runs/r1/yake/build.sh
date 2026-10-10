#!/bin/sh
set -e
cd "$(dirname "$0")"
mojo build myake/kernel.mojo --emit shared-lib -o myake/_kernel.so
