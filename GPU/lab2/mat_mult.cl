__kernel void
matrixMul(__global float* in1, 
          __global float* in2, 
          __global float* output, 
          int N){
	int x = get_global_id(0);
	int y = get_global_id(1);
	
	float value = 0;
	for (int k=0; k<N; ++k) {
		float a = in1[k + N * y];
		float b = in2[k*N +x];
		value += a * b;
	}
	output[x + N * y] = value;
}
