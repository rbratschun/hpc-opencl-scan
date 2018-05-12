__kernel 
void scan_exclusive_sequential(__global const int * restrict input, __global int * output,const  int n) {
  output[0] = 0;
  for (int i=1; i<n; i++) {
    output[i] = output[i-1] + input[i-1];
  }
}

#define WARP_SHIFT 4
#define GRP_SHIFT 8
#define BANK_OFFSET(n)     ((n) >> WARP_SHIFT + (n) >> GRP_SHIFT)
__kernel 
void scan_parallel_naive(__global const int * restrict input, __global int * output, const int n)
{
    __global int* temp = output;
    int pout = 1, pin = 0;

    // load input into temp
    // This is exclusive scan, so shift right by one and set first elt to 0
    int thread = get_local_id(0);
    temp[pout*n + thread] = (thread > 0) ? input[thread-1] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pout;
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[pout*n+thread] = temp[pin*n+thread];
        if (thread >= offset)
            temp[pout*n+thread] += temp[pin*n+thread - offset];
    }

    output[thread] = temp[pout*n+thread]; // write output
}

int upsweep(__local int * temp, int offset, int n)
{
	int lid = get_local_id(0);
        for (int d = n>>1; d > 0; d >>= 1)
        {   
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < d)
            {  
                int ai = offset * (2*lid + 1)-1;
                int bi = offset * (2*lid + 2)-1;
                ai += BANK_OFFSET(ai);
                bi += BANK_OFFSET(bi);
                temp[bi] += temp[ai];  
            }  
            offset <<= 1; 
        }
		return offset;
}

void downsweep(__local int * temp, int offset, int n)
{
	int lid = get_local_id(0);
	for (int d = 1; d < n; d <<= 1)
		{  
			offset >>= 1;  
			barrier(CLK_LOCAL_MEM_FENCE);

			if (lid < d)
			{
				int ai = offset * (2*lid + 1)-1;
				int bi = offset * (2*lid + 2)-1;
				ai += BANK_OFFSET(ai);
				bi += BANK_OFFSET(bi);
		  
				int t = temp[ai];  
				temp[ai] = temp[bi];  
				temp[bi] += t;   
			}
		}
}

__kernel void scan_blelloch(__global const int* input,
                            __global int* output,
							__local int * temp,
                            const int block_size)
{
    int lid = get_local_id(0);
    int group_id = get_group_id(0);
    int n = get_local_size(0) * 2;
 
    int group_offset = group_id * block_size;
    int maxval = 0;
    do
    {
        // calculate array indices and offsets to avoid SLM bank conflicts
        int ai = lid;
        int bi = lid + (n>>1);  
        int bankOffsetA = BANK_OFFSET(ai);
        int bankOffsetB = BANK_OFFSET(bi);
 
        // load input into local memory
        temp[ai + bankOffsetA] = input[group_offset + ai];
        temp[bi + bankOffsetB] = input[group_offset + bi];
		
        // parallel prefix sum up sweep phase
		int offset = upsweep(temp, 1, n);
        // clear the last element
        if (lid == 0)
        {
            temp[n - 1 + BANK_OFFSET(n - 1)] = 0;
        }
 
        // down sweep phase
        downsweep(temp, offset, n);  
		
        barrier(CLK_LOCAL_MEM_FENCE);
        //output scan result to global memory
        output[group_offset + ai] = temp[ai + bankOffsetA] + maxval;
        output[group_offset + bi] = temp[bi + bankOffsetB] + maxval;
 
        //update cumulative prefix sum and shift offset for next iteration
        maxval += temp[n - 1 + BANK_OFFSET(n - 1)] + input[group_offset + n - 1];
        group_offset += n;
    }
    while(group_offset < (group_id + 1) * block_size);
}
