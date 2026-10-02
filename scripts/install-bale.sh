#!/usr/bin/env bash
# ==============================================================================
# Hermes Agent - Bale Gateway One-Click Installer & Updater
# Repository: https://github.com/doodoolyasin/hermes-agent
# Branch: feat/bale-gateway-offline-mode
# ==============================================================================

set -eo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

INSTALL_DIR="${INSTALL_DIR:-/opt/hermes-agent}"
REPO_URL="${REPO_URL:-https://github.com/doodoolyasin/hermes-agent.git}"
BRANCH="${BRANCH:-feat/bale-gateway-offline-mode}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

BALE_TOKEN=""
BALE_USERS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token)
      BALE_TOKEN="$2"
      shift 2
      ;;
    --allowed-users)
      BALE_USERS="$2"
      shift 2
      ;;
    --dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    *)
      echo -e "${RED}گزینه نامعتبر:${NC} $1"
      exit 1
      ;;
  esac
done

echo ""
echo -e "${CYAN}${BOLD}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║       نصب و راه‌اندازی خودکار هرمس برای پیام‌رسان بله       ║${NC}"
echo -e "${CYAN}${BOLD}║       Hermes Agent - Bale Gateway Auto Installer      ║${NC}"
echo -e "${CYAN}${BOLD}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Check Root / Sudo
SUDO=""
if [[ $EUID -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo -e "${RED}خطا: برای نصب بسته‌های سیستمی نیاز به دسترسی root یا sudo است.${NC}"
    exit 1
  fi
fi

# 2. Get credentials if not provided
if [[ -z "$BALE_TOKEN" ]]; then
  # Check if existing token exists
  if [[ -f "$HERMES_HOME/.env" ]]; then
    EXISTING_TOKEN=$(grep -E '^BALE_BOT_TOKEN=' "$HERMES_HOME/.env" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
    if [[ -n "$EXISTING_TOKEN" ]]; then
      echo -e "${YELLOW}توکن فعلی بله در سیستم پیدا شد: ${EXISTING_TOKEN:0:8}...${NC}"
      read -rp "آیا می‌خواهید از همین توکن استفاده کنید؟ [Y/n]: " USE_EXISTING
      if [[ "$USE_EXISTING" =~ ^[Nn] ]]; then
        read -rp "لطفاً توکن ربات بله (از BotFather بله) را وارد کنید: " BALE_TOKEN
      else
        BALE_TOKEN="$EXISTING_TOKEN"
      fi
    fi
  fi
fi

if [[ -z "$BALE_TOKEN" ]]; then
  read -rp "لطفاً توکن ربات بله (از BotFather بله) را وارد کنید: " BALE_TOKEN
  if [[ -z "$BALE_TOKEN" ]]; then
    echo -e "${RED}خطا: توکن ربات بله الزامی است.${NC}"
    exit 1
  fi
fi

if [[ -z "$BALE_USERS" ]]; then
  if [[ -f "$HERMES_HOME/.env" ]]; then
    EXISTING_USERS=$(grep -E '^BALE_ALLOWED_USERS=' "$HERMES_HOME/.env" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
    if [[ -n "$EXISTING_USERS" ]]; then
      BALE_USERS="$EXISTING_USERS"
    fi
  fi
fi

if [[ -z "$BALE_USERS" ]]; then
  read -rp "شناسه کاربری عددی شما در بله (اختیاری - برای امنیت): " BALE_USERS
fi

# 3. Verify Bale Token
echo -e "${CYAN}→${NC} در حال بررسی و اعتبارسنجی توکن بله..."
ME_RESP=$(curl -s -m 10 "https://tapi.bale.ai/bot${BALE_TOKEN}/getMe" || true)
if [[ "$ME_RESP" =~ \"ok\":true ]]; then
  BOT_NAME=$(echo "$ME_RESP" | grep -o '"username":"[^"]*' | cut -d'"' -f4 || echo "Bot")
  echo -e "${GREEN}✓${NC} توکن معتبر است. ربات متصل: ${BOLD}@${BOT_NAME}${NC}"
else
  echo -e "${YELLOW}هشدار: ارتباط مستقیم با API بله برقرار نشد یا توکن نادرست است. نصب ادامه می‌یابد.${NC}"
fi

# 4. System packages
echo -e "${CYAN}→${NC} در حال به‌روزرسانی و نصب بسته‌های مورد نیاز لینوکس..."
if command -v apt-get >/dev/null 2>&1; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq python3 python3-venv python3-pip git curl build-essential systemd
elif command -v dnf >/dev/null 2>&1; then
  $SUDO dnf install -y -q python3 python3-pip git curl gcc systemd
elif command -v pacman >/dev/null 2>&1; then
  $SUDO pacman -Sy --noconfirm python python-pip git curl base-devel systemd
fi

# 5. Clone or Update Repo
echo -e "${CYAN}→${NC} در حال دریافت آخرین نسخه کدهای هرمس از گیت‌هاب..."
$SUDO mkdir -p "$INSTALL_DIR"
$SUDO chown -R "$(id -u):$(id -g)" "$INSTALL_DIR"

if [[ -d "$INSTALL_DIR/.git" ]]; then
  cd "$INSTALL_DIR"
  git remote set-url origin "$REPO_URL" 2>/dev/null || true
  git fetch origin "$BRANCH" -q
  git checkout "$BRANCH" -q
  git pull origin "$BRANCH" -q
  echo -e "${GREEN}✓${NC} ریپازیتوری با موفقیت به‌روزرسانی شد."
else
  git clone -b "$BRANCH" --depth 1 "$REPO_URL" "$INSTALL_DIR" -q
  cd "$INSTALL_DIR"
  echo -e "${GREEN}✓${NC} کدهای هرمس کلون شدند."
fi

# 6. Python Environment Setup
echo -e "${CYAN}→${NC} در حال راه‌اندازی محیط مجازی پایتون و وابستگی‌ها..."
if [[ ! -d "$INSTALL_DIR/venv" ]]; then
  python3 -m venv "$INSTALL_DIR/venv"
fi

"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
"$INSTALL_DIR/venv/bin/pip" install -e "$INSTALL_DIR" -q
"$INSTALL_DIR/venv/bin/pip" install -q httpx openai ruamel.yaml prompt_toolkit rich croniter python-dotenv tenacity pydantic

# Symlink CLI
$SUDO ln -sf "$INSTALL_DIR/venv/bin/hermes" /usr/local/bin/hermes
echo -e "${GREEN}✓${NC} دستور ${BOLD}hermes${NC} در سیستم ثبت شد."

# 7. Write Configuration
echo -e "${CYAN}→${NC} در حال تنظیم کانفیگ‌های بله و پرووایدر هوش مصنوعی..."
mkdir -p "$HERMES_HOME"

cat << EOF > "$HERMES_HOME/.env"
BALE_BOT_TOKEN="${BALE_TOKEN}"
BALE_ALLOWED_USERS="${BALE_USERS}"
EOF

chmod 600 "$HERMES_HOME/.env"

# Preserve existing config or write standard config
cat << EOF > "$HERMES_HOME/config.yaml"
# تنظیمات پیام‌رسان بله
bale:
  token: "${BALE_TOKEN}"
  allowed_users: "${BALE_USERS}"
  markdown: true

# فعال‌کردن پلتفرم بله در گیت‌وی
gateway:
  platforms:
    - bale

# فال‌بک خودکار به ارائه‌دهنده‌های رایگان هوش مصنوعی در صورت نبود API Key
free_providers:
  enabled: true
  auto_discover: true

agent:
  max_turns: 150

telemetry:
  shared_metrics:
    enabled: false
    send: false
EOF

# 8. Setup Systemd Service
echo -e "${CYAN}→${NC} در حال ثبت و فعال‌سازی سرویس دائمی Background (Systemd)..."
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$USER_SYSTEMD_DIR"

# Enable linger so service survives logout
if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "$USER" 2>/dev/null || true
fi

cat << EOF > "$USER_SYSTEMD_DIR/hermes-gateway.service"
[Unit]
Description=Hermes Agent Gateway - Messaging Platform Integration
After=network.target

[Service]
Type=simple
WorkingDirectory=${HERMES_HOME}
Environment=HERMES_HOME=${HERMES_HOME}
ExecStart=${INSTALL_DIR}/venv/bin/hermes gateway run
Restart=always
RestartSec=5
KillMode=control-group
LimitNOFILE=65536

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable hermes-gateway
systemctl --user restart hermes-gateway

# 9. Verification
sleep 3
echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}✓ نصب و فعال‌سازی هرمس برای بله با موفقیت به پایان رسید!${NC}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "🔹 ${BOLD}وضعیت سرویس گیت‌وی:${NC}"
systemctl --user status hermes-gateway --no-pager | head -n 10
echo ""
echo -e "💡 ${BOLD}دستورات کاربردی:${NC}"
echo -e "  - بررسی وضعیت: ${CYAN}hermes gateway status${NC}"
echo -e "  - لاگ‌های زنده: ${CYAN}hermes gateway logs -f${NC}"
echo -e "  - ری‌استارت سرویس: ${CYAN}hermes gateway restart${NC}"
echo ""
echo -e "🤖 هم‌اکنون به ربات خود در بله پیام دهید تا پاسخ دهد."
echo ""
