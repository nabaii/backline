#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

# Install Node.js if not available
if ! command -v node &> /dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

cd frontend
npm install
npm run build
