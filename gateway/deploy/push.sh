#!/usr/bin/env bash
# Сборка llm-gateway на этой машине под linux x86_64 и выкладка на VPS.
#   SSH="ssh -i ~/.ssh/hello user@91.188.212.172" gateway/deploy/push.sh
# Первый раз дополнительно: SETUP=1 — подготовка сервера (deploy/setup.sh).
# Нужны: rustup target add x86_64-unknown-linux-musl, zig, cargo install cargo-zigbuild.
set -euo pipefail

: "${SSH:?задайте SSH, например SSH=\"ssh -i ~/.ssh/hello user@host\"}"
cd "$(dirname "$0")/../.."
TARGET=x86_64-unknown-linux-musl

cargo zigbuild --release -p llm-gateway --target "$TARGET"
[ -f gateway/gateway.toml ] || { echo "нет gateway/gateway.toml — скопируйте gateway.example.toml и задайте ключи"; exit 1; }

# Всё нужное — одним tar-потоком в ~/llm на сервере.
stage=$(mktemp -d)
mkdir -p "$stage/deploy"
cp gateway/deploy/* "$stage/deploy/"
cp "target/$TARGET/release/llm-gateway" gateway/gateway.toml "$stage/"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$stage" -cf - . | $SSH 'mkdir -p ~/llm && tar -xf - -C ~/llm'
rm -rf "$stage"

[ "${SETUP:-0}" = 1 ] && $SSH 'bash ~/llm/deploy/setup.sh'

$SSH 'set -e
sudo install -o llm -g llm -m 755 ~/llm/llm-gateway /opt/llm/gateway/llm-gateway
sudo install -o root -g llm -m 640 ~/llm/gateway.toml /etc/llm/gateway.toml
sudo cp ~/llm/deploy/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llm-gateway >/dev/null
sudo systemctl restart llm-gateway
sleep 1
systemctl --no-pager --lines=5 status llm-gateway'
