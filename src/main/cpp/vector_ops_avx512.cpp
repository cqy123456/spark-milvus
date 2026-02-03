#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <jni.h>

// AVX-512 specific includes
#ifdef __AVX512F__
#include <avx512fintrin.h>
#endif
#ifdef __AVX512CD__
#include <avx512cdintrin.h>
#endif

// CPU feature detection using cpuid
static bool has_avx512() {
    static bool checked = false;
    static bool result = false;
    
    if (!checked) {
        #ifdef __GNUC__
            // Use GCC builtin if available (requires calling __builtin_cpu_init first)
            __builtin_cpu_init();
            #if defined(__AVX512F__) && defined(__AVX512CD__)
                result = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512cd");
            #else
                result = false;
            #endif
        #else
            // For other compilers or if builtin not available, use compile-time check
            // Runtime will handle gracefully if CPU doesn't support it
            #if defined(__AVX512F__) && defined(__AVX512CD__)
                result = true;  // Assume support if compiled with AVX-512 flags
            #else
                result = false;
            #endif
        #endif
        checked = true;
    }
    return result;
}

static bool has_avx2() {
    static bool checked = false;
    static bool result = false;
    
    if (!checked) {
        #ifdef __GNUC__
            __builtin_cpu_init();
            #if defined(__AVX2__) && defined(__FMA__)
                result = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
            #else
                result = false;
            #endif
        #else
            #if defined(__AVX2__) && defined(__FMA__)
                result = true;
            #else
                result = false;
            #endif
        #endif
        checked = true;
    }
    return result;
}

// Helper for AVX2 horizontal sum (used by both AVX-512 and AVX2 code)
static inline float _hsum8_ps(__m256 x) {
    __m128 low = _mm256_extractf128_ps(x, 0);
    __m128 high = _mm256_extractf128_ps(x, 1);
    low = _mm_add_ps(low, high);
    __m128 shuf = _mm_shuffle_ps(low, low, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(low, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// AVX-512 optimized dot product
#ifdef __AVX512F__
static float dot_product_avx512(const float* a, const float* b, int n) {
    const int simd_width = 16; // 512 bits / 32 bits
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    
    // Process 16 elements at a time
    for (; i + simd_width <= n; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum: reduce 512-bit register to scalar
    // Split into two 256-bit halves
    __m256 sum_low = _mm512_extractf32x8_ps(sum, 0);
    __m256 sum_high = _mm512_extractf32x8_ps(sum, 1);
    __m256 sum256 = _mm256_add_ps(sum_low, sum_high);
    // Manual horizontal sum for AVX256
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}
#else
static float dot_product_avx512(const float* a, const float* b, int n) {
    // Not compiled with AVX-512, fallback to AVX2 or scalar
    return dot_product_scalar(a, b, n);
}
#endif

// AVX2 optimized dot product (fallback)
#ifdef __AVX2__
static float dot_product_avx2(const float* a, const float* b, int n) {
    const int simd_width = 8; // 256 bits / 32 bits
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    
    for (; i + simd_width <= n; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Manual horizontal sum for AVX256
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}
#else
static float dot_product_avx2(const float* a, const float* b, int n) {
    return dot_product_scalar(a, b, n);
}
#endif

// Pure scalar dot product (fallback)
static float dot_product_scalar(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// AVX-512 optimized L2 norm
#ifdef __AVX512F__
static float norm2_avx512(const float* a, int n) {
    const int simd_width = 16;
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    
    for (; i + simd_width <= n; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        sum = _mm512_fmadd_ps(va, va, sum);
    }
    
    // Horizontal sum: reduce 512-bit register to scalar
    __m256 sum_low = _mm512_extractf32x8_ps(sum, 0);
    __m256 sum_high = _mm512_extractf32x8_ps(sum, 1);
    __m256 sum256 = _mm256_add_ps(sum_low, sum_high);
    // Manual horizontal sum for AVX256
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    for (; i < n; i++) {
        result += a[i] * a[i];
    }
    
    return std::sqrt(result);
}
#else
static float norm2_avx512(const float* a, int n) {
    return norm2_scalar(a, n);
}
#endif

// AVX2 optimized L2 norm (fallback)
#ifdef __AVX2__
static float norm2_avx2(const float* a, int n) {
    const int simd_width = 8;
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    
    for (; i + simd_width <= n; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        sum = _mm256_fmadd_ps(va, va, sum);
    }
    
    // Manual horizontal sum for AVX256
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    for (; i < n; i++) {
        result += a[i] * a[i];
    }
    
    return std::sqrt(result);
}
#else
static float norm2_avx2(const float* a, int n) {
    return norm2_scalar(a, n);
}
#endif

// Pure scalar L2 norm (fallback)
static float norm2_scalar(const float* a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * a[i];
    }
    return std::sqrt(sum);
}

// Function pointers for optimized implementations
static float (*dot_product_func)(const float*, const float*, int) = nullptr;
static float (*norm2_func)(const float*, int) = nullptr;

static void init_functions() {
    static bool initialized = false;
    if (initialized) return;
    
    // Try AVX-512 first (if compiled with support and CPU supports it)
    #ifdef __AVX512F__
    if (has_avx512()) {
        dot_product_func = dot_product_avx512;
        norm2_func = norm2_avx512;
        initialized = true;
        return;
    }
    #endif
    
    // Fallback to AVX2 (if compiled with support and CPU supports it)
    #ifdef __AVX2__
    if (has_avx2()) {
        dot_product_func = dot_product_avx2;
        norm2_func = norm2_avx2;
        initialized = true;
        return;
    }
    #endif
    
    // Final fallback: scalar implementation
    dot_product_func = dot_product_scalar;
    norm2_func = norm2_scalar;
    initialized = true;
}

extern "C" {

// JNI: Compute dot product
// Java signature: public static native float dotProduct(float[] a, float[] b, int n);
// OPTIMIZED: Uses GetPrimitiveArrayCritical to minimize memory copying
// This gives direct access to Java heap memory when possible, avoiding copies
JNIEXPORT jfloat JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_dotProduct(
    JNIEnv *env, jclass clazz,
    jfloatArray a, jfloatArray b, jint n) {
    
    init_functions();
    
    // Use GetPrimitiveArrayCritical for direct memory access (no copy if possible)
    // Must release quickly to avoid blocking GC
    jfloat* a_ptr = (jfloat*)env->GetPrimitiveArrayCritical(a, nullptr);
    if (a_ptr == nullptr) return 0.0f;  // Out of memory
    
    jfloat* b_ptr = (jfloat*)env->GetPrimitiveArrayCritical(b, nullptr);
    if (b_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);
        return 0.0f;
    }
    
    float result = dot_product_func(a_ptr, b_ptr, n);
    
    // Release immediately after computation (critical for GC)
    env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(b, b_ptr, JNI_ABORT);
    
    return result;
}

// JNI: Compute L2 norm
// Java signature: public static native float norm2(float[] a, int n);
// OPTIMIZED: Uses GetPrimitiveArrayCritical for direct memory access
JNIEXPORT jfloat JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_norm2(
    JNIEnv *env, jclass clazz,
    jfloatArray a, jint n) {
    
    init_functions();
    
    jfloat* a_ptr = (jfloat*)env->GetPrimitiveArrayCritical(a, nullptr);
    if (a_ptr == nullptr) return 0.0f;
    
    float result = norm2_func(a_ptr, n);
    
    env->ReleasePrimitiveArrayCritical(a, a_ptr, JNI_ABORT);
    
    return result;
}

// JNI: Find nearest targets (optimized batch version)
// Java signature: public static native void findNearestTargets(float[] queries, float[] targets,
//     int[] nearestIndices, float[] nearestDistances, int nQueries, int nTargets, int dim);
JNIEXPORT void JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_findNearestTargets(
    JNIEnv *env, jclass clazz,
    jfloatArray queries, jfloatArray targets,
    jintArray nearestIndices, jfloatArray nearestDistances,
    jint nQueries, jint nTargets, jint dim) {
    
    init_functions();
    
    // Use GetPrimitiveArrayCritical for direct memory access (minimize copying)
    jfloat* queries_ptr = (jfloat*)env->GetPrimitiveArrayCritical(queries, nullptr);
    if (queries_ptr == nullptr) return;
    
    jfloat* targets_ptr = (jfloat*)env->GetPrimitiveArrayCritical(targets, nullptr);
    if (targets_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(queries, queries_ptr, JNI_ABORT);
        return;
    }
    
    jint* indices_ptr = (jint*)env->GetPrimitiveArrayCritical(nearestIndices, nullptr);
    if (indices_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(queries, queries_ptr, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(targets, targets_ptr, JNI_ABORT);
        return;
    }
    
    jfloat* distances_ptr = (jfloat*)env->GetPrimitiveArrayCritical(nearestDistances, nullptr);
    if (distances_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(queries, queries_ptr, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(targets, targets_ptr, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(nearestIndices, indices_ptr, JNI_ABORT);
        return;
    }
    
    // Pre-compute target norms squared
    float* targetNormsSq = new float[nTargets];
    for (int t = 0; t < nTargets; t++) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            float val = targets_ptr[t * dim + d];
            sum += val * val;
        }
        targetNormsSq[t] = sum;
    }
    
    // Process each query
    for (int q = 0; q < nQueries; q++) {
        const float* query = queries_ptr + q * dim;
        
        // Compute query norm squared
        float queryNormSq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float val = query[d];
            queryNormSq += val * val;
        }
        
        float minDist = std::numeric_limits<float>::max();
        int minIdx = 0;
        
        // Find nearest target
        for (int t = 0; t < nTargets; t++) {
            const float* target = targets_ptr + t * dim;
            float dot = dot_product_func(query, target, dim);
            float dist = std::max(0.0f, queryNormSq + targetNormsSq[t] - 2.0f * dot);
            
            if (dist < minDist) {
                minDist = dist;
                minIdx = t;
            }
        }
        
        indices_ptr[q] = minIdx;
        distances_ptr[q] = minDist;
    }
    
    delete[] targetNormsSq;
    
    // Release immediately after computation (critical for GC)
    env->ReleasePrimitiveArrayCritical(queries, queries_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(targets, targets_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(nearestIndices, indices_ptr, 0);  // 0 = copy back changes
    env->ReleasePrimitiveArrayCritical(nearestDistances, distances_ptr, 0);  // 0 = copy back changes
}

// JNI: Compute pairwise squared distances
// Java signature: public static native void computePairwiseSquaredDistances(...)
JNIEXPORT void JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_computePairwiseSquaredDistances(
    JNIEnv *env, jclass clazz,
    jfloatArray queries, jfloatArray targets, jfloatArray result,
    jint nQueries, jint nTargets, jint dim) {
    
    init_functions();
    
    // Use GetPrimitiveArrayCritical for direct memory access
    jfloat* queries_ptr = (jfloat*)env->GetPrimitiveArrayCritical(queries, nullptr);
    if (queries_ptr == nullptr) return;
    
    jfloat* targets_ptr = (jfloat*)env->GetPrimitiveArrayCritical(targets, nullptr);
    if (targets_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(queries, queries_ptr, JNI_ABORT);
        return;
    }
    
    jfloat* result_ptr = (jfloat*)env->GetPrimitiveArrayCritical(result, nullptr);
    if (result_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(queries, queries_ptr, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(targets, targets_ptr, JNI_ABORT);
        return;
    }
    
    // Compute pairwise squared distances
    for (int q = 0; q < nQueries; q++) {
        const float* query = queries_ptr + q * dim;
        float queryNormSq = norm2_func(query, dim);
        queryNormSq = queryNormSq * queryNormSq;  // squared norm
        
        for (int t = 0; t < nTargets; t++) {
            const float* target = targets_ptr + t * dim;
            float targetNormSq = norm2_func(target, dim);
            targetNormSq = targetNormSq * targetNormSq;  // squared norm
            
            float dot = dot_product_func(query, target, dim);
            
            // Squared Euclidean distance = ||a||^2 + ||b||^2 - 2*a·b
            float distSq = queryNormSq + targetNormSq - 2.0f * dot;
            result_ptr[q * nTargets + t] = distSq;
        }
    }
    
    // Release immediately after computation
    env->ReleasePrimitiveArrayCritical(queries, queries_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(targets, targets_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(result, result_ptr, 0);  // 0 = copy back changes
}

// JNI: Batch Euclidean distance
// Java signature: public static native void batchEuclideanDistance(float[] query, float[] targets,
//     float[] distances, int nTargets, int dim);
JNIEXPORT void JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_batchEuclideanDistance(
    JNIEnv *env, jclass clazz,
    jfloatArray query, jfloatArray targets,
    jfloatArray distances, jint nTargets, jint dim) {
    
    init_functions();
    
    // Use GetPrimitiveArrayCritical for direct memory access
    jfloat* query_ptr = (jfloat*)env->GetPrimitiveArrayCritical(query, nullptr);
    if (query_ptr == nullptr) return;
    
    jfloat* targets_ptr = (jfloat*)env->GetPrimitiveArrayCritical(targets, nullptr);
    if (targets_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(query, query_ptr, JNI_ABORT);
        return;
    }
    
    jfloat* distances_ptr = (jfloat*)env->GetPrimitiveArrayCritical(distances, nullptr);
    if (distances_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(query, query_ptr, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(targets, targets_ptr, JNI_ABORT);
        return;
    }
    
    float queryNorm = norm2_func(query_ptr, dim);
    float queryNormSq = queryNorm * queryNorm;
    
    for (int t = 0; t < nTargets; t++) {
        const float* target = targets_ptr + t * dim;
        float targetNorm = norm2_func(target, dim);
        float targetNormSq = targetNorm * targetNorm;
        
        float dot = dot_product_func(query_ptr, target, dim);
        
        // Squared Euclidean distance = ||a||^2 + ||b||^2 - 2*a·b
        float distSq = queryNormSq + targetNormSq - 2.0f * dot;
        distances_ptr[t] = std::sqrt(distSq);  // Return actual distance, not squared
    }
    
    // Release immediately after computation
    env->ReleasePrimitiveArrayCritical(query, query_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(targets, targets_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(distances, distances_ptr, 0);  // 0 = copy back changes
}

// JNI: Batch cosine distance
// Java signature: public static native void batchCosineDistance(float[] query, float[] targets,
//     float[] distances, int nTargets, int dim);
JNIEXPORT void JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_batchCosineDistance(
    JNIEnv *env, jclass clazz,
    jfloatArray query, jfloatArray targets,
    jfloatArray distances, jint nTargets, jint dim) {

    init_functions();

    // Use GetPrimitiveArrayCritical for direct memory access
    jfloat* query_ptr = (jfloat*)env->GetPrimitiveArrayCritical(query, nullptr);
    if (query_ptr == nullptr) return;

    jfloat* targets_ptr = (jfloat*)env->GetPrimitiveArrayCritical(targets, nullptr);
    if (targets_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(query, query_ptr, JNI_ABORT);
        return;
    }

    jfloat* distances_ptr = (jfloat*)env->GetPrimitiveArrayCritical(distances, nullptr);
    if (distances_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(query, query_ptr, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(targets, targets_ptr, JNI_ABORT);
        return;
    }

    float queryNorm = norm2_func(query_ptr, dim);

    for (int t = 0; t < nTargets; t++) {
        const float* target = targets_ptr + t * dim;
        float dot = dot_product_func(query_ptr, target, dim);
        float targetNorm = norm2_func(target, dim);

        float cosineSim = (queryNorm > 0 && targetNorm > 0) ? (dot / (queryNorm * targetNorm)) : 0.0f;
        distances_ptr[t] = 1.0f - cosineSim;
    }

    // Release immediately after computation
    env->ReleasePrimitiveArrayCritical(query, query_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(targets, targets_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(distances, distances_ptr, 0);  // 0 = copy back changes
}

// ==================== DirectByteBuffer versions (true zero-copy) ====================
// These functions use GetDirectBufferAddress which provides direct pointer access
// to off-heap memory with NO copying at all.

// JNI: Compute dot product using DirectByteBuffer (zero-copy)
// Java signature: public static native float dotProductDirect(ByteBuffer a, ByteBuffer b, int n);
JNIEXPORT jfloat JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_dotProductDirect(
    JNIEnv *env, jclass clazz,
    jobject a, jobject b, jint n) {

    init_functions();

    // GetDirectBufferAddress returns direct pointer to off-heap memory (true zero-copy)
    float* a_ptr = (float*)env->GetDirectBufferAddress(a);
    float* b_ptr = (float*)env->GetDirectBufferAddress(b);

    if (a_ptr == nullptr || b_ptr == nullptr) {
        // Not a direct buffer or JVM doesn't support this
        return 0.0f;
    }

    return dot_product_func(a_ptr, b_ptr, n);
}

// JNI: Compute L2 norm using DirectByteBuffer (zero-copy)
// Java signature: public static native float norm2Direct(ByteBuffer a, int n);
JNIEXPORT jfloat JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_norm2Direct(
    JNIEnv *env, jclass clazz,
    jobject a, jint n) {

    init_functions();

    float* a_ptr = (float*)env->GetDirectBufferAddress(a);

    if (a_ptr == nullptr) {
        return 0.0f;
    }

    return norm2_func(a_ptr, n);
}

// JNI: Find nearest targets using DirectByteBuffer (zero-copy for vectors)
// Java signature: public static native void findNearestTargetsDirect(ByteBuffer queries, ByteBuffer targets,
//     int[] nearestIndices, float[] nearestDistances, int nQueries, int nTargets, int dim);
JNIEXPORT void JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_findNearestTargetsDirect(
    JNIEnv *env, jclass clazz,
    jobject queries, jobject targets,
    jintArray nearestIndices, jfloatArray nearestDistances,
    jint nQueries, jint nTargets, jint dim) {

    init_functions();

    // Zero-copy access to vector data
    float* queries_ptr = (float*)env->GetDirectBufferAddress(queries);
    float* targets_ptr = (float*)env->GetDirectBufferAddress(targets);

    if (queries_ptr == nullptr || targets_ptr == nullptr) {
        return;
    }

    // Output arrays still need GetPrimitiveArrayCritical
    jint* indices_ptr = (jint*)env->GetPrimitiveArrayCritical(nearestIndices, nullptr);
    if (indices_ptr == nullptr) return;

    jfloat* distances_ptr = (jfloat*)env->GetPrimitiveArrayCritical(nearestDistances, nullptr);
    if (distances_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(nearestIndices, indices_ptr, JNI_ABORT);
        return;
    }

    // Pre-compute target norms squared
    float* targetNormsSq = new float[nTargets];
    for (int t = 0; t < nTargets; t++) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            float val = targets_ptr[t * dim + d];
            sum += val * val;
        }
        targetNormsSq[t] = sum;
    }

    // Process each query
    for (int q = 0; q < nQueries; q++) {
        const float* query = queries_ptr + q * dim;

        // Compute query norm squared
        float queryNormSq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float val = query[d];
            queryNormSq += val * val;
        }

        float minDist = std::numeric_limits<float>::max();
        int minIdx = 0;

        // Find nearest target
        for (int t = 0; t < nTargets; t++) {
            const float* target = targets_ptr + t * dim;
            float dot = dot_product_func(query, target, dim);
            float dist = std::max(0.0f, queryNormSq + targetNormsSq[t] - 2.0f * dot);

            if (dist < minDist) {
                minDist = dist;
                minIdx = t;
            }
        }

        indices_ptr[q] = minIdx;
        distances_ptr[q] = minDist;
    }

    delete[] targetNormsSq;

    env->ReleasePrimitiveArrayCritical(nearestIndices, indices_ptr, 0);
    env->ReleasePrimitiveArrayCritical(nearestDistances, distances_ptr, 0);
}

// JNI: Batch compute squared L2 distances for pairs using DirectByteBuffer (zero-copy)
// Java signature: public static native void batchPairDistancesDirect(ByteBuffer vectors,
//     int[] pairsI, int[] pairsJ, float[] distances, int nPairs, int nVectors, int dim);
JNIEXPORT void JNICALL
Java_com_zilliz_spark_connector_utils_VectorOpsNative_batchPairDistancesDirect(
    JNIEnv *env, jclass clazz,
    jobject vectors,
    jintArray pairsI, jintArray pairsJ,
    jfloatArray distances,
    jint nPairs, jint nVectors, jint dim) {

    init_functions();

    // Zero-copy access to vector data
    float* vectors_ptr = (float*)env->GetDirectBufferAddress(vectors);
    if (vectors_ptr == nullptr) return;

    // Get pair indices
    jint* pairsI_ptr = (jint*)env->GetPrimitiveArrayCritical(pairsI, nullptr);
    if (pairsI_ptr == nullptr) return;

    jint* pairsJ_ptr = (jint*)env->GetPrimitiveArrayCritical(pairsJ, nullptr);
    if (pairsJ_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(pairsI, pairsI_ptr, JNI_ABORT);
        return;
    }

    jfloat* distances_ptr = (jfloat*)env->GetPrimitiveArrayCritical(distances, nullptr);
    if (distances_ptr == nullptr) {
        env->ReleasePrimitiveArrayCritical(pairsI, pairsI_ptr, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(pairsJ, pairsJ_ptr, JNI_ABORT);
        return;
    }

    // Pre-compute squared norms for all vectors
    float* normsSq = new float[nVectors];
    for (int v = 0; v < nVectors; v++) {
        float sum = 0.0f;
        const float* vec = vectors_ptr + v * dim;
        for (int d = 0; d < dim; d++) {
            sum += vec[d] * vec[d];
        }
        normsSq[v] = sum;
    }

    // Compute distances for each pair
    for (int p = 0; p < nPairs; p++) {
        int i = pairsI_ptr[p];
        int j = pairsJ_ptr[p];

        const float* vi = vectors_ptr + i * dim;
        const float* vj = vectors_ptr + j * dim;

        float dot = dot_product_func(vi, vj, dim);

        // L2 squared: ||a - b||² = ||a||² + ||b||² - 2·a·b
        distances_ptr[p] = std::max(0.0f, normsSq[i] + normsSq[j] - 2.0f * dot);
    }

    delete[] normsSq;

    env->ReleasePrimitiveArrayCritical(pairsI, pairsI_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(pairsJ, pairsJ_ptr, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(distances, distances_ptr, 0);
}

} // extern "C"
