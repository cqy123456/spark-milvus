#!/bin/bash
# Build and upload native library to S3
#
# Note: For local development, just run `sbt compile` or `sbt assembly`.
#       The sbt build will automatically compile and package the native library.
#
# This script is for manually uploading to S3 (e.g., for deployment).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
S3_BUCKET="s3://vdc-cloud-dev/SessionStage/job-02f46718f29vfqaxrsuvjw/cqy-dedup/native-libs/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Native Library Build & S3 Upload${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Build the library
echo -e "${YELLOW}[1/2] Building native library...${NC}"
cd "${SCRIPT_DIR}"

if [ ! -f "build.sh" ]; then
    echo -e "${RED}ERROR: build.sh not found in ${SCRIPT_DIR}${NC}"
    exit 1
fi

chmod +x build.sh
./build.sh

if [ ! -f "${BUILD_DIR}/libvectorops_avx512.so" ]; then
    echo -e "${RED}ERROR: Build failed - library not found${NC}"
    exit 1
fi

LIB_SIZE=$(du -h "${BUILD_DIR}/libvectorops_avx512.so" | cut -f1)
echo -e "${GREEN}✓ Build successful: ${BUILD_DIR}/libvectorops_avx512.so (${LIB_SIZE})${NC}"
echo ""

# Step 2: Upload to S3
echo -e "${YELLOW}[2/2] Uploading to S3...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${YELLOW}WARNING: aws CLI not found, skipping S3 upload${NC}"
    echo -e "${YELLOW}To upload manually, run:${NC}"
    echo "  aws s3 cp ${BUILD_DIR}/libvectorops_avx512.so ${S3_BUCKET}libvectorops_avx512.so"
    exit 0
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${YELLOW}WARNING: AWS credentials not configured, skipping S3 upload${NC}"
    echo -e "${YELLOW}To configure, run: aws configure${NC}"
    exit 0
fi

echo "Uploading to: ${S3_BUCKET}libvectorops_avx512.so"
aws s3 cp "${BUILD_DIR}/libvectorops_avx512.so" "${S3_BUCKET}libvectorops_avx512.so"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Upload successful${NC}"
    echo -e "${GREEN}Library available at: ${S3_BUCKET}libvectorops_avx512.so${NC}"
else
    echo -e "${RED}ERROR: Upload failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Note: For local development, use 'sbt compile' or 'sbt assembly'."
echo "      The sbt build will automatically compile and package the native library."
echo ""
