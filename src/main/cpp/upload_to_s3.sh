#!/bin/bash
# Upload native library to S3

set -e

LIB_PATH="/home/ubuntu/spark-milvus/src/main/cpp/build/libvectorops_avx512.so"
S3_BUCKET="s3://vdc-cloud-dev/SessionStage/job-02f46718f29vfqaxrsuvjw/cqy-dedup/native-libs/"

if [ ! -f "$LIB_PATH" ]; then
    echo "ERROR: Library not found at $LIB_PATH"
    echo "Please build the library first: cd src/main/cpp && ./build.sh"
    exit 1
fi

echo "Uploading native library to S3..."
echo "Source: $LIB_PATH"
echo "Destination: ${S3_BUCKET}libvectorops_avx512.so"

aws s3 cp "$LIB_PATH" "${S3_BUCKET}libvectorops_avx512.so"

echo "Upload complete!"
echo "Library available at: ${S3_BUCKET}libvectorops_avx512.so"
