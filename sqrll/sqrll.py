import cupy
import torch
import math

# This is the whole algorithm.
# Everything below is just optimizations of this function
def naive_sqrll(x, r, prev=None):
    y = []
    if prev is None:
        prev = torch.zeros_like(x[:,0])
    for i in range(x.shape[1]):
        prev = prev * r[:,i] + x[:,i]
        y += [prev]
    return torch.stack(y, dim=1)


_CODE = '''
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template<class T>
struct starray
{
    T * ptr;
    int st;
    T & operator[](int i) { return ptr[i * st]; }
};

// we assume y is initialized to x
// and that y[t=0] includes y[t=-1] * r[t=0]
template<typename T>
__global__ void sqrll_forward(
    T * y, T const* r, 
    int s0, int s1, int s2,
    int yst0, int yst1, int yst2,
    int rst0, int rst1, int rst2)
{
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int i2 = blockIdx.y * blockDim.y + threadIdx.y;
    if(i0 >= s0 || i2 >= s2) { return; }

    auto Y = starray{y + i0 * yst0 + i2 * yst2, yst1};
    auto R = starray{r + i0 * rst0 + i2 * rst2, rst1};

    for(int i1 = 1; i1 < s1; i1++)
    {
        Y[i1] += Y[i1-1] * R[i1];
    }
}

// xg and rg must be c-contiguous
template<typename T>
__global__ void sqrll_backward(
    T * xg, T * rg,
    T const* yg, T const* y, T const* r, 
    int s0, int s1, int s2,
    int ygst0, int ygst1, int ygst2,
    int yst0, int yst1, int yst2,
    int rst0, int rst1, int rst2)
{
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int i2 = blockIdx.y * blockDim.y + threadIdx.y;
    if(i0 >= s0 || i2 >= s2) { return; }

    int st0 = s1 * s2;
    int st1 = s2;
    int st2 = 1;
    auto XG = starray{xg + i0 * st0 + i2 * st2, st1};
    auto RG = starray{rg + i0 * st0 + i2 * st2, st1};
    auto YG = starray{yg + i0 * ygst0 + i2 * ygst2, ygst1};
    auto Y = starray{y + i0 * yst0 + i2 * yst2, yst1};
    auto R = starray{r + i0 * rst0 + i2 * rst2, rst1};

    XG[s1-1] = YG[s1-1];
    RG[s1-1] = Y[s1-2] * YG[s1-1];
    for(int i1 = s1-2; i1 > 0; i1--)
    {
        XG[i1] = YG[i1] + R[i1+1] * XG[i1+1];
        RG[i1] = Y[i1-1] * XG[i1];
    }
    XG[0] = YG[0] + R[1] * XG[1];
    RG[0] = 0; // set outside the kernel
}

'''


class SqrllKernelCuda(torch.autograd.Function):
    module = None
    type_map = {
        torch.float32: 'float',
        torch.float16: '__half',
        torch.bfloat16: '__nv_bfloat16',
    }

    @staticmethod
    def prep(ctx):
        if SqrllKernelCuda.module is None:
            types = list(SqrllKernelCuda.type_map.values())
            name_exp = [f'sqrll_backward<{n}>' for n in types]
            name_exp += [f'sqrll_forward<{n}>' for n in types]
            SqrllKernelCuda.module = cupy.RawModule(
                code=_CODE, 
                options=('-std=c++20',), 
                name_expressions=name_exp,
            )

    @staticmethod
    def forward(ctx, x, r, prev=None):
        SqrllKernelCuda.prep(ctx)
        T = x.shape[1]
        assert T>=2, "not implemented for T < 2 steps"

        y = x.detach().clone()
        if prev is not None:
            y[:,0] += prev * r[:,0]


        thrd = (1, min(512, y.shape[2]))
        blkx = math.ceil(y.shape[0] / thrd[0])
        blky = math.ceil(y.shape[2] / thrd[1])
        kernel = SqrllKernelCuda.module.get_function(
            f'sqrll_forward<{SqrllKernelCuda.type_map[y.dtype]}>')
        

        kernel((blkx, blky), thrd, (
            y.data_ptr(), r.data_ptr(),
            *y.shape,
            *[y.stride(i) for i in range(3)],
            *[r.stride(i) for i in range(3)],
        ))

        ctx.save_for_backward(prev, r, y)

        return y

    @staticmethod
    def backward(ctx, yg):
        SqrllKernelCuda.prep(ctx)
        T = yg.shape[1]
        assert T>=2, "not implemented for T < 2 steps"

        prev, r, y = ctx.saved_tensors
        
        rg = torch.empty_like(r).contiguous()
        xg = torch.empty_like(r).contiguous()

        thrd = (1, min(512, y.shape[2]))
        blkx = math.ceil(y.shape[0] / thrd[0])
        blky = math.ceil(y.shape[2] / thrd[1])
        kernel = SqrllKernelCuda.module.get_function(
            f'sqrll_backward<{SqrllKernelCuda.type_map[y.dtype]}>')

        kernel((blkx, blky), thrd, (
            xg.data_ptr(), rg.data_ptr(),
            yg.data_ptr(), y.data_ptr(), r.data_ptr(),
            *y.shape,
            *[yg.stride(i) for i in range(3)],
            *[y.stride(i) for i in range(3)],
            *[r.stride(i) for i in range(3)],
        ))

        if prev is not None:
            rg[:,0] = prev * xg[:,0]
            prevg = r[:,0] * xg[:,0]
        else:
            prevg = None

        return xg, rg, prevg
    

@torch.fx.wrap
def sqrll_kernel(x, r, prev=None):
    if x.is_cuda:
        return SqrllKernelCuda.apply(x, r, prev)
    else:
        return naive_sqrll(x, r, prev)