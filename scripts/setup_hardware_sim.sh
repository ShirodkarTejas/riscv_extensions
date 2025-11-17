#!/bin/bash
# Host-side launcher for Chipyard hardware simulation setup
# Run this from your Mac/Linux host

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Sparse Attention Hardware Simulation Setup (Docker)         ║"
echo "╔════════════════════════════════════════════════════════════════╗"
echo ""
echo "This script will:"
echo "  1. Build the Chipyard Docker image (~15 min)"
echo "  2. Start the container"
echo "  3. Run automated Chipyard setup inside container (~70 min)"
echo ""
echo "Total time: ~90 minutes (mostly automated)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Step 1: Build Docker image (with Chipyard dependencies)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Building Docker image..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker-compose -f docker/docker-compose.yml build

echo "✅ Docker image built"
echo ""

# Step 2: Start container
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Starting container..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if container already running
if docker ps | grep -q sattn_rvv_dev; then
    echo "ℹ️  Container already running"
else
    docker-compose -f docker/docker-compose.yml up -d
    echo "✅ Container started"
fi

echo ""
sleep 2

# Step 3: Run setup inside container
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Running Chipyard setup (this will take ~70 minutes)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⏰ Started at: $(date)"
echo ""

docker exec -it sattn_rvv_dev bash /workspace/scripts/setup_chipyard_in_docker.sh

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Hardware Simulation Ready! ✅                     ║"
echo "╔════════════════════════════════════════════════════════════════╗"
echo ""
echo "⏰ Completed at: $(date)"
echo ""
echo "🎉 Chipyard is now set up with the sparse attention accelerator!"
echo ""
echo "📝 Next steps:"
echo ""
echo "  1. Enter the container:"
echo "     docker exec -it sattn_rvv_dev bash"
echo ""
echo "  2. Run tests:"
echo "     cd /workspace/chipyard/sims/verilator"
echo "     ./simulator-chipyard-SattnDebugConfig /workspace/hw/sim/test_hw_basic"
echo ""
echo "  3. Or use Python interface:"
echo "     python3 /workspace/python/hardware_simulator.py"
echo ""
echo "📚 Documentation:"
echo "  - Setup guide: hw/DOCKER_CHIPYARD_SETUP.md"
echo "  - Phase 3 plan: hw/PHASE3_HARDWARE_PLAN.md"
echo "  - Quick start: hw/CHIPYARD_QUICKSTART.md"
echo ""

