// matrix convolution
__kernel void Convolve(
    const __global unsigned char * in, 
    __global unsigned char * out, 
    const int dataSizeX, 
    const int dataSizeY, 
    const __global float * filter, 
    const int kernelSizeX, 
    const int kernelSizeY)
{
    int i, j, m, n, mm, nn;
    int kCenterX, kCenterY;
    int rowIndex, colIndex;
    float sum = 0.0f;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;

    i = get_global_id(0);
    j = get_global_id(1);

    for (m = 0; m < kernelSizeX; m++)
    {
        mm = kernelSizeY - 1 - m;
        
        for (n = 0; n < kernelSizeY; n++)
        {
            nn = kernelSizeX - 1 - n;

            rowIndex = i + m - kCenterY;
            colIndex = j + n - kCenterX;

            // ignore input samples which are out of bound
            if (rowIndex >= 0 &&
                rowIndex < dataSizeY &&
                colIndex >= 0 &&
                colIndex < dataSizeX)
            {
                sum += in[dataSizeX * rowIndex + colIndex] * filter[kernelSizeX * mm + nn];
            }
            
            // sum += in[dataSizeX * rowIndex + colIndex] * filter[kernelSizeX * mm + nn];
        }
    }

    out[dataSizeX * i + j] = (unsigned char)((float)fabs(sum) + 0.5f);
}
