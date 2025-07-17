
template<class scalar_t>
__global__ void rope_kernel(
        scalar_t* rotated_queries, const scalar_t* queries, const float* cosines, const float* sines,
        int F, int W, int Hq, int S, int E, int RotaryE)
{
    int f = blockIdx.x / S;
    int s = blockIdx.x % S;
    int h = blockIdx.y;
    int w = blockIdx.z;

    const scalar_t* query = queries + ((w * Hq + h) * S + s) * E;
    scalar_t* result = rotated_queries + (((f * W + w) * Hq + h) * S + s) * E;
    int e = threadIdx.x;

    if (e < RotaryE / 2) {
        float x1 = query[e];
        float x2 = query[e + RotaryE / 2];

        // fetch a tuple of activations, which we imagine as a complex number
        int offset = (((f*W + w) * S + s) * RotaryE);

        result[e] = x1 * cosines[offset + e] - x2 * sines[offset + e];
        result[e + RotaryE / 2] = x2 * cosines[offset + e + RotaryE / 2] + x1 * sines[offset + e + RotaryE / 2];
    } else { // This thread handles a pair of non-rotated dimensions
        // e is in [RotaryE/2, E/2)
        result[e] = query[e];
        result[e + E/2] = query[e + E/2];
    }
}

template<class scalar_t>
void rope_gpu(
        scalar_t* rotated_queries, const scalar_t* queries, const float* cosines, const float* sines,
        int F, int W, int Hq, int S, int E, int RotaryE) {
    dim3 grid_dim(F*S , Hq, W);
    dim3 block_dim(E/2, 1, 1);
    rope_kernel<<<grid_dim, block_dim>>>(rotated_queries, queries, cosines, sines, F, W, Hq, S, E, RotaryE);
}
