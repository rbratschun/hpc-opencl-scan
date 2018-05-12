__kernel 
void scan_exclusive_sequential(__global const int * restrict input, __global int * output,const  int n) {
  output[0] = 0;
  for (int i=1; i<n; i++) {
    output[i] = output[i-1] + input[i-1];
  }
}