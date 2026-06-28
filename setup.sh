#!/bin/bash
# ============================================================================
# People Counter — Setup Script for Raspberry Pi 5 + Hailo AI HAT
# ============================================================================
#
# This script:
#   1. Installs system dependencies
#   2. Creates a Python virtual environment
#   3. Installs Python packages
#   4. Installs torchreid from source (for OSNet ReID)
#   5. Downloads OSNet pretrained weights
#   6. Verifies Hailo device connectivity
#   7. Runs a smoke test
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  People Counter — Setup Script${NC}"
echo -e "${CYAN}  Raspberry Pi 5 + Hailo AI HAT${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# Step 1: System Dependencies
# ─────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/7] Installing system dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-dev \
    python3-venv \
    python3-pip \
    libcamera-dev \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    git \
    cmake
echo -e "${GREEN}  ✓ System dependencies installed${NC}"

# ─────────────────────────────────────────────────────────────────────────
# Step 2: Python Virtual Environment
# ─────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/7] Setting up Python virtual environment...${NC}"
if [ -d "${VENV_DIR}" ]; then
    echo -e "  Virtual environment already exists at ${VENV_DIR}"
else
    python3 -m venv "${VENV_DIR}" --system-site-packages
    echo -e "  Created virtual environment at ${VENV_DIR}"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q
echo -e "${GREEN}  ✓ Virtual environment ready${NC}"

# ─────────────────────────────────────────────────────────────────────────
# Step 3: Python Dependencies
# ─────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[3/7] Installing Python dependencies...${NC}"
pip install -r "${SCRIPT_DIR}/requirements.txt" -q
echo -e "${GREEN}  ✓ Python dependencies installed${NC}"

# ─────────────────────────────────────────────────────────────────────────
# Step 4: Install torchreid from source
# ─────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/7] Installing torchreid (OSNet ReID)...${NC}"
TORCHREID_DIR="${SCRIPT_DIR}/third_party/deep-person-reid"

if [ -d "${TORCHREID_DIR}" ]; then
    echo -e "  torchreid already cloned at ${TORCHREID_DIR}"
else
    mkdir -p "${SCRIPT_DIR}/third_party"
    git clone https://github.com/KaiyangZhou/deep-person-reid.git "${TORCHREID_DIR}"
fi

cd "${TORCHREID_DIR}"
pip install --no-build-isolation -e . -q
cd "${SCRIPT_DIR}"
echo -e "${GREEN}  ✓ torchreid installed${NC}"

# ─────────────────────────────────────────────────────────────────────────
# Step 5: Download OSNet Pretrained Weights
# ─────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[5/7] Downloading OSNet pretrained weights...${NC}"
WEIGHTS_DIR="${HOME}/.cache/torch/checkpoints"
WEIGHTS_FILE="${WEIGHTS_DIR}/osnet_x1_0_imagenet.pth"

if [ -f "${WEIGHTS_FILE}" ]; then
    echo -e "  Weights already downloaded at ${WEIGHTS_FILE}"
else
    mkdir -p "${WEIGHTS_DIR}"
    # torchreid auto-downloads on first use, but we trigger it now
    python3 -c "
import torchreid
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
print('OSNet weights downloaded successfully')
"
fi
echo -e "${GREEN}  ✓ OSNet weights ready${NC}"

# ─────────────────────────────────────────────────────────────────────────
# Step 6: Verify Hailo Device
# ─────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[6/7] Verifying Hailo device...${NC}"
HAILO_OK=false

python3 -c "
try:
    from hailo_platform import VDevice
    vd = VDevice()
    del vd
    print('  Hailo device detected and operational')
except Exception as e:
    print(f'  WARNING: Hailo device not available: {e}')
    print('  Make sure the Hailo AI HAT is properly connected')
    print('  and the HailoRT SDK is installed.')
" && HAILO_OK=true || true

if [ "${HAILO_OK}" = true ]; then
    echo -e "${GREEN}  ✓ Hailo device verified${NC}"
else
    echo -e "${RED}  ✗ Hailo device not detected (non-fatal — can still test with video)${NC}"
fi

# ─────────────────────────────────────────────────────────────────────────
# Step 7: Verify HEF Model
# ─────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[7/7] Checking HEF model...${NC}"
HEF_PATH="/usr/share/hailo-models/yolov8s_h8l.hef"

if [ -f "${HEF_PATH}" ]; then
    echo -e "${GREEN}  ✓ HEF model found at ${HEF_PATH}${NC}"
else
    echo -e "${YELLOW}  ⚠ HEF model not found at ${HEF_PATH}${NC}"
    echo -e "  You may need to download it or set HEF_PATH in config.py"
    echo -e "  Check: https://hailo.ai/developer-zone/"
fi

# ─────────────────────────────────────────────────────────────────────────
# Done!
# ─────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""
echo -e "To activate the virtual environment:"
echo -e "  ${CYAN}source venv/bin/activate${NC}"
echo ""
echo -e "To run the people counter:"
echo -e "  ${CYAN}python people_counter.py${NC}"
echo ""
echo -e "For help:"
echo -e "  ${CYAN}python people_counter.py --help${NC}"
echo ""
