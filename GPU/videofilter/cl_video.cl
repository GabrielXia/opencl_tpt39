__kernel void
matrixMul(__global float* in1, 
          __global float* in2, 
          __global float* output, 
          int row_s){
	int y = get_global_id(0);
	
	float value = 0;
	for (int k=0; k<row_s; ++k) {
		float a = in1[k + row_s * y];
		float b = in2[k];
		value += a * b;
	}
	output[y] = value;
};
