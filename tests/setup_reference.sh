#!/bin/sh
# The reference stack the parity tests compare against: OneJev's qev runtime at the commit elephant-1 ships for,
# with the checkpoint's serve.patch applied. Creates ./OneJev in the current directory.
set -e
git clone -q https://github.com/OmniJev/OneJev.git
git -C OneJev checkout -q 81ce62f1597c91e46767d4d02d6ac2e18534fe94
curl -sfL https://huggingface.co/baselabs/elephant-1/raw/main/serve.patch -o serve.patch
git -C OneJev apply ../serve.patch
echo "OneJev ready at $(pwd)/OneJev"
