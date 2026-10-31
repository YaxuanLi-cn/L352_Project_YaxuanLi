import numpy as np
import torch
import cupy as cp
_COMMON = r"""
// 空间哈希里乘在 y 坐标上的质数（x 坐标乘的是 1，不用写出来）；这组质数出自 Instant-NGP
#define PRIME_Y 2654435761u
// 空间哈希里乘在 z 坐标上的质数
#define PRIME_Z 805459861u
// 一个采样点在某一层网格里的位置
struct Cell {
    // 点所在格子的整数坐标（格子八个角里坐标最小的那个角）
    unsigned int ix, iy, iz;
    // 点在格子里的小数位置（0 到 1），三线性插值的权重由它得到
    float wx, wy, wz;
};
// 求第 n 个点在分辨率为 r 的那一层里落在哪个格子、在格子里的什么位置；x 是 [N,3] 按行展开的坐标数组，取值 [0,1]
__device__ inline Cell locate(const float* x, int n, float r) {
    // 要返回的结果
    Cell c;
    // 点的三个坐标乘上分辨率，变成“以格子为单位”的坐标
    const float px = x[3 * n] * r, py = x[3 * n + 1] * r, pz = x[3 * n + 2] * r;
    // 向下取整，得到所在格子的坐标
    const float fx = floorf(px), fy = floorf(py), fz = floorf(pz);
    // 小数部分就是点在格子里的相对位置
    c.wx = px - fx; c.wy = py - fy; c.wz = pz - fz;
    // 格子坐标先转成 int 再转成无符号整数；后面的哈希用无符号运算（乘法溢出时自动按 2^32 取模）
    c.ix = (unsigned int)(int)fx; c.iy = (unsigned int)(int)fy; c.iz = (unsigned int)(int)fz;
    // 返回格子坐标和小数位置
    return c;
}
// 求格子的第 k 个角点（k = 0..7）在这一层特征表里的行号，并通过 w 返回它的三线性插值权重。
// 角点的顺序和 Python 里的 itertools.product((0, 1), repeat=3) 相同：x 是变化最慢的那一位，z 是变化最快的那一位
__device__ inline unsigned int corner_index(const Cell& c, int k, unsigned int s, bool dense,
                                            // mask = 每层表的行数 - 1（行数是 2 的幂）；w 用来把权重传出去
                                            unsigned int mask, float* w) {
    // 从 k 的三个二进制位取出这个角点在 x、y、z 方向上要不要加 1
    const unsigned int dx = (k >> 2) & 1, dy = (k >> 1) & 1, dz = k & 1;
    // 角点的整数坐标
    const unsigned int cx = c.ix + dx, cy = c.iy + dy, cz = c.iz + dz;
    // 三线性插值的权重：每个方向上，加了 1 的角点用小数位置，没加的用 1 减小数位置，三个方向相乘
    *w = (dx ? c.wx : 1.f - c.wx) * (dy ? c.wy : 1.f - c.wy) * (dz ? c.wz : 1.f - c.wz);
    // 粗的层（整层的角点数不超过表的行数）：每个角点有自己单独的一行，没有哈希冲突
    if (dense) {
        // 把三维坐标展开成一维下标；s 是每个方向上的角点数（分辨率 + 2）
        const unsigned int idx = cx + cy * s + cz * s * s;
        // 超出表的范围就截到最后一行（PyTorch 版本里对应的是 clamp）
        return idx > mask ? mask : idx;
    }
    // 细的层：三个坐标分别乘上质数（1、PRIME_Y、PRIME_Z）再按位异或，然后和 mask 按位与，相当于对表的行数取模
    return (cx ^ (cy * PRIME_Y) ^ (cz * PRIME_Z)) & mask;
}
"""
_FORWARD = _COMMON + r"""
// 前向 kernel：每个线程负责一个（点, 层）组合，把格子 8 个角点的特征按三线性权重加起来。
// extern "C" 让函数名不被 C++ 改编，CuPy 才能按名字找到它；__global__ 表示这是可以从主机端启动的 kernel
extern "C" __global__
// 参数：x 是输入坐标 [N,3]，table 是特征表 [L*T,2]（__restrict__ 告诉编译器这些指针指向的内存互不重叠）
void hashgrid_forward(const float* __restrict__ x, const float* __restrict__ table,
                      // res 是各层的分辨率 [L]，stride 是各层每个方向上的角点数 [L]
                      const float* __restrict__ res, const int* __restrict__ stride,
                      // dense 标明各层是否稠密存储 [L]，out 是输出 [N, L*2]
                      const int* __restrict__ dense, float* __restrict__ out,
                      // N 是点数，L 是层数，T 是每层表的行数
                      const int N, const int L, const int T)
{
    // 这个线程的全局编号，也就是第几个（点, 层）组合；用 64 位整数，免得超出 int 的范围
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    // 线程数是按每块 256 个向上取整的，多出来的线程没有任务，直接返回
    if (i >= (long long)N * L) return;
    // 拆成点的编号 n 和层的编号 l
    const int n = (int)(i / L), l = (int)(i % L);
    // 这个点在第 l 层落在哪个格子、在格子里的什么位置
    const Cell c = locate(x, n, res[l]);
    // T 是 2 的幂，T - 1 当掩码，按位与就是对 T 取模
    const unsigned int mask = (unsigned int)(T - 1);
    // 第 l 层的特征在整张表里的起始位置（每行 2 个特征）
    const float* tab = table + (size_t)l * T * 2;
    // 两个特征各自的累加结果
    float o0 = 0.f, o1 = 0.f;
    // 遍历格子的 8 个角点
    for (int k = 0; k < 8; ++k) {
        // 这个角点的插值权重
        float w;
        // 这个角点在表里的行号（同时得到权重 w）
        const unsigned int idx = corner_index(c, k, (unsigned int)stride[l], dense[l] != 0, mask, &w);
        // 第 1 个特征按权重累加
        o0 += w * tab[2 * idx];
        // 第 2 个特征按权重累加
        o1 += w * tab[2 * idx + 1];
    }
    // 写进输出：第 n 个点第 l 层的第 1 个特征在 out 的第 2*i 个位置
    out[2 * i] = o0;
    // 第 2 个特征紧跟在后面
    out[2 * i + 1] = o1;
}
"""
_BACKWARD = _COMMON + r"""
// 反向 kernel：每个线程负责一个（点, 层）组合，把这个输出的梯度按同样的三线性权重加到 8 个角点对应的表项上。
// extern "C" 让函数名不被 C++ 改编；__global__ 表示这是可以从主机端启动的 kernel
extern "C" __global__
// 参数：x 是输入坐标 [N,3]，grad_out 是损失对前向输出的梯度 [N, L*2]
void hashgrid_backward(const float* __restrict__ x, const float* __restrict__ grad_out,
                       // res 是各层的分辨率 [L]，stride 是各层每个方向上的角点数 [L]
                       const float* __restrict__ res, const int* __restrict__ stride,
                       // dense 标明各层是否稠密存储 [L]，grad_table 是要累加出来的特征表梯度 [L*T,2]（调用前已清零）
                       const int* __restrict__ dense, float* __restrict__ grad_table,
                       // N 是点数，L 是层数，T 是每层表的行数
                       const int N, const int L, const int T)
{
    // 这个线程的全局编号，也就是第几个（点, 层）组合
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    // 多出来的线程没有任务，直接返回
    if (i >= (long long)N * L) return;
    // 损失对这个（点, 层）的两个输出特征的梯度
    const float g0 = grad_out[2 * i], g1 = grad_out[2 * i + 1];
    // 两个梯度都是 0 时对表没有贡献，直接跳过，省时间
    if (g0 == 0.f && g1 == 0.f) return;
    // 拆成点的编号 n 和层的编号 l
    const int n = (int)(i / L), l = (int)(i % L);
    // 重新算出这个点在第 l 层的格子和小数位置（和前向时一样）
    const Cell c = locate(x, n, res[l]);
    // 对表的行数取模用的掩码
    const unsigned int mask = (unsigned int)(T - 1);
    // 第 l 层的梯度在整张梯度表里的起始位置
    float* tab = grad_table + (size_t)l * T * 2;
    // 遍历格子的 8 个角点
    for (int k = 0; k < 8; ++k) {
        // 这个角点的插值权重
        float w;
        // 这个角点在表里的行号（同时得到权重 w）
        const unsigned int idx = corner_index(c, k, (unsigned int)stride[l], dense[l] != 0, mask, &w);
        // 前向时 输出 = sum(w * 表项)，所以表项的梯度 = w * 输出的梯度；很多线程可能同时加到同一个表项上，要用原子加法
        atomicAdd(tab + 2 * idx, w * g0);
        // 第 2 个特征同样处理
        atomicAdd(tab + 2 * idx + 1, w * g1);
    }
}
"""
_kernels = {}


def _kernel(name):
    if name not in _kernels:
        src = _FORWARD if name == "hashgrid_forward" else _BACKWARD
        _kernels[name] = cp.RawKernel(src, name)
    return _kernels[name]


def _launch(name, n, levels, *args):
    total = n * levels
    block = 256
    _kernel(name)(((total + block - 1) // block,), (block,), args)


class _HashGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, res, stride, dense, table_size):
        n, levels = x.shape[0], res.shape[0]
        x = x.detach().contiguous()
        out = torch.empty((n, levels * 2), dtype=torch.float32, device=x.device)
        if n > 0:
            _launch("hashgrid_forward", n, levels, cp.asarray(x), cp.asarray(table.detach()), cp.asarray(res),
                    cp.asarray(stride), cp.asarray(dense), cp.asarray(out),
                    np.int32(n), np.int32(levels), np.int32(table_size))
        ctx.save_for_backward(x, res, stride, dense)
        ctx.table_shape = table.shape
        ctx.table_size = table_size
        return out
    @staticmethod
    def backward(ctx, grad_out):
        x, res, stride, dense = ctx.saved_tensors
        n, levels = x.shape[0], res.shape[0]
        grad_table = torch.zeros(ctx.table_shape, dtype=torch.float32, device=x.device)
        if n > 0:
            g = grad_out.detach().float().contiguous()
            _launch("hashgrid_backward", n, levels, cp.asarray(x), cp.asarray(g), cp.asarray(res),
                    cp.asarray(stride), cp.asarray(dense), cp.asarray(grad_table),
                    np.int32(n), np.int32(levels), np.int32(ctx.table_size))
        return None, grad_table, None, None, None, None


def hashgrid(x, table, res, stride, dense, table_size):
    return _HashGrid.apply(x.float(), table, res, stride, dense, table_size)
if __name__ == "__main__":
    import time
    from .nerf import HashEncoding
    torch.manual_seed(0)
    for kw in (dict(num_levels=16, min_res=16, max_res=2048, log2_hashmap_size=19),
               dict(num_levels=5, min_res=16, max_res=128, log2_hashmap_size=17)):
        enc = HashEncoding(**kw).cuda()
        with torch.no_grad():
            enc.table.normal_()
        x = torch.rand(200_000, 3, device="cuda")
        x[:10] = torch.tensor([0.0, 1.0, 0.5], device="cuda")
        a = enc._forward_torch(x)
        b = enc._forward_cuda(x)
        wgt = torch.randn_like(a)
        ga, = torch.autograd.grad((a * wgt).sum(), enc.table)
        gb, = torch.autograd.grad((b * wgt).sum(), enc.table)
        print(f"levels={kw['num_levels']:2d}  max |out diff| = {(a - b).abs().max().item():.2e}  "
              f"max |grad diff| = {(ga - gb).abs().max().item():.2e}  (|grad| max {ga.abs().max().item():.2f})")
        x = torch.rand(4096 * 256, 3, device="cuda")
        for name, fn in (("torch", enc._forward_torch), ("cuda", enc._forward_cuda)):
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(5):
                out = fn(x)
                out.sum().backward()
            torch.cuda.synchronize()
            print(f"    {name:5s} forward+backward, 1M points: {(time.time() - t) / 5 * 1e3:.1f} ms")
