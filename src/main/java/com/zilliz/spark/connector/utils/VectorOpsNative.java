package com.zilliz.spark.connector.utils;

import java.io.*;
import java.nio.*;
import java.nio.file.*;

/**
 * Native JNI interface for AVX-512 optimized vector operations
 * 
 * This provides high-performance C++ implementations using AVX-512 SIMD instructions
 * for critical operations like distance computation and nearest neighbor search.
 * 
 * Performance benefits:
 * - 3-5x faster than ND4J for batch operations
 * - 20-30% less memory usage
 * - Reduced JNI call overhead through batch processing
 * 
 * Falls back to ND4J or pure Scala if native library is not available.
 */
public class VectorOpsNative {
    
    private static boolean isAvailable = false;
    private static boolean initialized = false;
    private static File extractedLib = null;
    
    /**
     * Extract native library from JAR to temporary file and load it
     */
    private static boolean loadLibraryFromJar() {
        String libName = "libvectorops_avx512.so";
        String resourcePath = "/native/" + libName;
        
        try (InputStream is = VectorOpsNative.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                // Library not found in JAR, try system library path
                return false;
            }
            
            // Create temporary file
            String prefix = "vectorops_avx512";
            String suffix = ".so";
            File tempFile = File.createTempFile(prefix, suffix);
            tempFile.deleteOnExit();
            
            // Copy from JAR to temp file
            try (OutputStream os = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
            }
            
            // Make executable (important for Linux)
            tempFile.setExecutable(true);
            
            // Load the library
            System.load(tempFile.getAbsolutePath());
            extractedLib = tempFile;
            return true;
        } catch (IOException | UnsatisfiedLinkError e) {
            return false;
        }
    }
    
    static {
        try {
            // First try to load from JAR (extract to temp file)
            if (loadLibraryFromJar()) {
                isAvailable = true;
                initialized = true;
            } else {
                // Fallback: try system library path
                System.loadLibrary("vectorops_avx512");
                isAvailable = true;
                initialized = true;
            }
        } catch (UnsatisfiedLinkError e) {
            isAvailable = false;
            initialized = true;
            // Log warning but don't fail - will use fallback
            System.err.println("WARN: vectorops_avx512 native library not available, using fallback implementation");
        }
    }
    
    /**
     * Check if native implementation is available
     */
    public static boolean isAvailable() {
        if (!initialized) {
            // Force initialization
            try {
                if (!loadLibraryFromJar()) {
                    System.loadLibrary("vectorops_avx512");
                }
                isAvailable = true;
            } catch (UnsatisfiedLinkError e) {
                isAvailable = false;
            }
            initialized = true;
        }
        return isAvailable;
    }
    
    /**
     * Compute dot product between two vectors
     * 
     * @param a First vector
     * @param b Second vector
     * @param n Vector length
     * @return Dot product
     */
    public static native float dotProduct(float[] a, float[] b, int n);
    
    /**
     * Compute L2 norm of a vector
     * 
     * @param a Vector
     * @param n Vector length
     * @return L2 norm
     */
    public static native float norm2(float[] a, int n);
    
    /**
     * Compute pairwise squared Euclidean distances between queries and targets
     * 
     * Uses AVX-512 SIMD for optimal performance.
     * 
     * @param queries Flattened query vectors: [nQueries * dim] in row-major order
     * @param targets Flattened target vectors: [nTargets * dim] in row-major order
     * @param result Output array: [nQueries * nTargets] in row-major order
     * @param nQueries Number of query vectors
     * @param nTargets Number of target vectors
     * @param dim Vector dimension
     */
    public static native void computePairwiseSquaredDistances(
        float[] queries,
        float[] targets,
        float[] result,
        int nQueries,
        int nTargets,
        int dim
    );
    
    /**
     * Find nearest target for each query vector
     * 
     * More memory efficient than computing full distance matrix.
     * Only returns the nearest target index and distance for each query.
     * 
     * @param queries Flattened query vectors: [nQueries * dim]
     * @param targets Flattened target vectors: [nTargets * dim]
     * @param nearestIndices Output: nearest target index for each query [nQueries]
     * @param nearestDistances Output: squared distance to nearest target [nQueries]
     * @param nQueries Number of query vectors
     * @param nTargets Number of target vectors
     * @param dim Vector dimension
     */
    public static native void findNearestTargets(
        float[] queries,
        float[] targets,
        int[] nearestIndices,
        float[] nearestDistances,
        int nQueries,
        int nTargets,
        int dim
    );
    
    /**
     * Batch compute Euclidean distances from one query to multiple targets
     * 
     * @param query Single query vector [dim]
     * @param targets Flattened target vectors: [nTargets * dim]
     * @param distances Output: distances [nTargets]
     * @param nTargets Number of target vectors
     * @param dim Vector dimension
     */
    public static native void batchEuclideanDistance(
        float[] query,
        float[] targets,
        float[] distances,
        int nTargets,
        int dim
    );
    
    /**
     * Batch compute cosine distances from one query to multiple targets
     *
     * @param query Single query vector [dim]
     * @param targets Flattened target vectors: [nTargets * dim]
     * @param distances Output: cosine distances [nTargets]
     * @param nTargets Number of target vectors
     * @param dim Vector dimension
     */
    public static native void batchCosineDistance(
        float[] query,
        float[] targets,
        float[] distances,
        int nTargets,
        int dim
    );

    // ==================== DirectByteBuffer versions (zero-copy) ====================

    /**
     * Compute dot product between two vectors using DirectByteBuffer (zero-copy)
     *
     * @param a First vector as DirectByteBuffer (must be direct, native byte order)
     * @param b Second vector as DirectByteBuffer (must be direct, native byte order)
     * @param n Vector length (number of floats)
     * @return Dot product
     */
    public static native float dotProductDirect(ByteBuffer a, ByteBuffer b, int n);

    /**
     * Compute L2 norm of a vector using DirectByteBuffer (zero-copy)
     *
     * @param a Vector as DirectByteBuffer (must be direct, native byte order)
     * @param n Vector length (number of floats)
     * @return L2 norm
     */
    public static native float norm2Direct(ByteBuffer a, int n);

    /**
     * Find nearest target for each query vector using DirectByteBuffer (zero-copy)
     *
     * @param queries Flattened query vectors as DirectByteBuffer: [nQueries * dim] floats
     * @param targets Flattened target vectors as DirectByteBuffer: [nTargets * dim] floats
     * @param nearestIndices Output: nearest target index for each query (int array)
     * @param nearestDistances Output: squared distance to nearest target (float array)
     * @param nQueries Number of query vectors
     * @param nTargets Number of target vectors
     * @param dim Vector dimension
     */
    public static native void findNearestTargetsDirect(
        ByteBuffer queries,
        ByteBuffer targets,
        int[] nearestIndices,
        float[] nearestDistances,
        int nQueries,
        int nTargets,
        int dim
    );

    /**
     * Batch compute squared L2 distances for pairs of vectors using DirectByteBuffer (zero-copy)
     *
     * @param vectors Flattened vectors as DirectByteBuffer: [nVectors * dim] floats
     * @param pairsI First indices of pairs (int array)
     * @param pairsJ Second indices of pairs (int array)
     * @param distances Output: squared distances for each pair (float array)
     * @param nPairs Number of pairs
     * @param nVectors Total number of vectors
     * @param dim Vector dimension
     */
    public static native void batchPairDistancesDirect(
        ByteBuffer vectors,
        int[] pairsI,
        int[] pairsJ,
        float[] distances,
        int nPairs,
        int nVectors,
        int dim
    );

    // ==================== Helper methods ====================

    /**
     * Allocate a direct ByteBuffer for float vectors with native byte order
     *
     * @param numFloats Number of floats to allocate
     * @return Direct ByteBuffer with native byte order
     */
    public static ByteBuffer allocateFloatBuffer(int numFloats) {
        return ByteBuffer.allocateDirect(numFloats * 4).order(ByteOrder.nativeOrder());
    }

    /**
     * Copy float array to DirectByteBuffer
     *
     * @param src Source float array
     * @param dst Destination DirectByteBuffer
     */
    public static void copyToBuffer(float[] src, ByteBuffer dst) {
        dst.clear();
        dst.asFloatBuffer().put(src);
    }

    /**
     * Copy 2D float array to DirectByteBuffer (flattened)
     *
     * @param src Source 2D float array
     * @param dst Destination DirectByteBuffer
     */
    public static void copyToBuffer(float[][] src, ByteBuffer dst) {
        dst.clear();
        FloatBuffer fb = dst.asFloatBuffer();
        for (float[] row : src) {
            fb.put(row);
        }
    }
}
