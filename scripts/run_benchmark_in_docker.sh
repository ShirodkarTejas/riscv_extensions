#!/bin/bash
# Run comprehensive benchmarks inside Docker container
#
# Usage:
#   ./scripts/run_benchmark_in_docker.sh [--L 128] [--D 32] [--patterns all]

set -e

# Default values
L=128
D=32
TECH_NODE="7nm"
PATTERNS="all"
OUTPUT="bench/results/comprehensive_docker_results.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --L)
      L="$2"
      shift 2
      ;;
    --D)
      D="$2"
      shift 2
      ;;
    --tech-node)
      TECH_NODE="$2"
      shift 2
      ;;
    --patterns)
      PATTERNS="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "================================================================================"
echo "Running Comprehensive Benchmarks in Docker"
echo "================================================================================"
echo "Problem size: L=$L, D=$D"
echo "Tech node: $TECH_NODE"
echo "Patterns: $PATTERNS"
echo "Output: $OUTPUT"
echo "================================================================================"
echo

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi

# Find existing container - try common names
CONTAINER_NAME=""
for name in "sattn_rvv_dev" "riscv_extensions_dev" "sattn-dev"; do
    if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
        CONTAINER_NAME="$name"
        echo "✅ Found running container: $CONTAINER_NAME"
        break
    fi
done

# If no running container found, check stopped containers
if [ -z "$CONTAINER_NAME" ]; then
    for name in "sattn_rvv_dev" "riscv_extensions_dev" "sattn-dev"; do
        if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
            CONTAINER_NAME="$name"
            echo "Starting existing container: $CONTAINER_NAME"
            docker start "$CONTAINER_NAME"
            break
        fi
    done
fi

# If still no container, error out with helpful message
if [ -z "$CONTAINER_NAME" ]; then
    echo "❌ Error: No suitable Docker container found."
    echo ""
    echo "Please start your container first, e.g.:"
    echo "  docker-compose up -d"
    echo ""
    echo "Or create a new one:"
    echo "  docker run -d --name sattn_rvv_dev \\"
    echo "    -v \$(pwd):/workspace \\"
    echo "    -w /workspace \\"
    echo "    sattn/rvv-dev:latest \\"
    echo "    sleep infinity"
    exit 1
fi

echo
echo "Executing benchmark inside container..."
echo

# Run the benchmark (no -t flag to avoid TTY issues in scripts)
docker exec "$CONTAINER_NAME" python3 bench/run_comprehensive_docker_benchmark.py \
    --L "$L" \
    --D "$D" \
    --tech-node "$TECH_NODE" \
    --patterns $PATTERNS \
    --output "$OUTPUT"

echo
echo "================================================================================"
echo "✅ Benchmark complete!"
echo "================================================================================"
echo "Results saved to: $OUTPUT"
echo
echo "Next steps:"
echo "  1. Generate unified comparison:"
echo "     python bench/create_unified_comparison.py --L $L --D $D"
echo
echo "  2. Visualize results:"
echo "     python bench/visualize_benchmarks.py --input $OUTPUT --all"
echo
echo "  3. Select optimal variant:"
echo "     python bench/variant_selector.py --max-memory-mb 1.0 --L $L --D $D"
echo "================================================================================"

