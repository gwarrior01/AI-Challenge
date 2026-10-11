#!/usr/bin/env bash
# Подготовка VPS (Ubuntu 24.04, только CPU) под локальную LLM: swap, llama.cpp,
# модели, systemd-юниты, firewall. Запускается на сервере от пользователя с sudo.
# Повторный запуск безопасен: уже сделанные шаги пропускаются.
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b11552}"
SWAP_SIZE="${SWAP_SIZE:-4G}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
ROOT=/opt/llm

# Модели: основная и запасная (меньше и быстрее). Какая работает — решает
# MODEL в /etc/llm/llama.env.
MODELS=(
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

echo "== swap"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l "$SWAP_SIZE" /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  # Свопить только при реальной нехватке памяти — веса модели должны жить в RAM.
  echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-llm.conf
  sudo sysctl -p /etc/sysctl.d/99-llm.conf
fi

echo "== пакеты"
# Готовая сборка llama.cpp линкуется с OpenMP.
dpkg -s libgomp1 >/dev/null 2>&1 || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgomp1; }

echo "== каталоги и системный пользователь"
id llm >/dev/null 2>&1 || sudo useradd --system --home "$ROOT" --shell /usr/sbin/nologin llm
sudo mkdir -p "$ROOT/llama" "$ROOT/models" "$ROOT/gateway" /etc/llm
sudo chown -R "$USER" "$ROOT"

echo "== llama.cpp $LLAMA_TAG"
if [ ! -x "$ROOT/llama/$LLAMA_TAG/llama-server" ]; then
  tmp=$(mktemp -d)
  curl -fsSL --retry 3 -o "$tmp/llama.tar.gz" \
    "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-ubuntu-x64.tar.gz"
  mkdir -p "$ROOT/llama/$LLAMA_TAG"
  tar -xzf "$tmp/llama.tar.gz" -C "$ROOT/llama/$LLAMA_TAG" --strip-components=1
  rm -rf "$tmp"
fi
ln -sfn "$ROOT/llama/$LLAMA_TAG" "$ROOT/llama/current"
"$ROOT/llama/current/llama-server" --version 2>&1 | tail -2

echo "== модели"
for url in "${MODELS[@]}"; do
  f="$ROOT/models/$(basename "$url")"
  [ -s "$f" ] || curl -fsSL --retry 3 -C - -o "$f" "$url"
done
ls -lh "$ROOT/models"

echo "== конфигурация"
if [ ! -f /etc/llm/llama.env ]; then
  sudo tee /etc/llm/llama.env >/dev/null <<EOF
# Модель и параметры llama-server. CTX_SIZE делится поровну между PARALLEL слотами.
MODEL=$ROOT/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
MODEL_ALIAS=qwen2.5-1.5b-instruct
PARALLEL=3
CTX_SIZE=6144
THREADS=$(nproc)
EOF
fi
sudo chown -R llm:llm "$ROOT"

echo "== systemd"
here="$(cd "$(dirname "$0")" && pwd)"
sudo cp "$here/llama-server.service" "$here/llm-gateway.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server

echo "== firewall: наружу только ssh и шлюз"
sudo ufw allow 22/tcp
sudo ufw allow "$GATEWAY_PORT/tcp"
sudo ufw --force enable
sudo ufw status
