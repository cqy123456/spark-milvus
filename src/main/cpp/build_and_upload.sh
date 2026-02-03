#!/bin/bash
# Build and upload native library script
# 
# This script:
# 1. Builds the C++ native library (libvectorops_avx512.so)
# 2. Copies it to resources directory for JAR packaging
# 3. Optionally uploads to S3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RESOURCES_DIR="${PROJECT_ROOT}/src/main/resources/native"
S3_BUCKET="s3://vdc-cloud-dev/SessionStage/job-02f46718f29vfqaxrsuvjw/cqy-dedup/native-libs/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Native Library Build & Upload Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Build the library
echo -e "${YELLOW}[1/3] Building native library...${NC}"
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

# Step 2: Copy to resources directory for JAR packaging
echo -e "${YELLOW}[2/3] Copying to resources directory for JAR packaging...${NC}"
mkdir -p "${RESOURCES_DIR}"

cp "${BUILD_DIR}/libvectorops_avx512.so" "${RESOURCES_DIR}/libvectorops_avx512.so"

if [ ! -f "${RESOURCES_DIR}/libvectorops_avx512.so" ]; then
    echo -e "${RED}ERROR: Failed to copy library to resources directory${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Copied to: ${RESOURCES_DIR}/libvectorops_avx512.so${NC}"
echo ""

# Step 3: Upload to S3 (optional)
echo -e "${YELLOW}[3/3] Uploading to S3...${NC}"

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
    echo ""
    echo -e "${GREEN}Library available at: ${S3_BUCKET}libvectorops_avx512.so${NC}"
else
    echo -e "${RED}ERROR: Upload failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build & Upload Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Rebuild JAR to include the native library:"
echo "   cd ${PROJECT_ROOT} && sbt clean assembly"
echo ""
echo "2. The library is now:"
echo "   - Built: ${BUILD_DIR}/libvectorops_avx512.so"
echo "   - In resources: ${RESOURCES_DIR}/libvectorops_avx512.so (will be packaged in JAR)"
echo "   - On S3: ${S3_BUCKET}libvectorops_avx512.so (backup/alternative)"
echo ""
