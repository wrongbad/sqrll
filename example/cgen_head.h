#include <memory>
#include <cassert>
#include <cmath>


// struct Mat
// {
//     float * ptr;
//     int h;
//     int w;
//     float & operator()(int i, int j) { return ptr[i*w + j]; }
// };

// struct Vec
// {
//     float * ptr;
//     int w;
//     float & operator()(int i) { return ptr[i]; }
//     float & operator[](int i) { return ptr[i]; }
// };


// void gemv(Vec l, Mat r, Vec out)
// {
//     constexpr int T = 16;

//     assert(l.w == r.h);
//     assert(r.w == out.w);
//     assert(r.w % T == 0);

//     for(int t = 0 ; t < r.w ; t += T)
//     {
//         for(int j = t ; j < t+T ; j++)
//             out[j] = 0;
//         for(int i = 0 ; i < r.h ; i++)
//             for(int j = t ; j < t+T ; j++)
//                 out[j] += l[i] * r(i,j);
//     }
// }

template<int S>
struct BFloat16Blob
{
    uint16_t data[S];
};

struct Zero {};


template<int H, int W>
struct Mat
{
    float data[H][W];
    
    Mat() {}
    Mat(Zero const&)
    {
        for(int i=0 ; i<H ; i++)
            for(int j=0 ; j<W; j++)
                data[i][j] = 0;
    }
    template<int S>
    Mat(BFloat16Blob<S> & blob, int offset)
    {
        for(int i=0 ; i<H ; i++)
            for(int j=0 ; j<W; j++)
            {
                int LE = 1; // not portable
                uint16_t * dst = reinterpret_cast<uint16_t*>(&data[i][j]);
                dst[LE] = blob.data[offset++];
            }
    }
    auto & operator()(int i) { return data[i]; }
    auto & operator[](int i) { return data[i]; }
    auto const& operator()(int i) const { return data[i]; }
    auto const& operator[](int i) const { return data[i]; }
};

template<int W>
struct Vec
{
    float data[W];
    
    Vec() {}
    Vec(Zero const&)
    {
        for(int j=0 ; j<W; j++)
            data[j] = 0;
    }
    template<int S>
    Vec(BFloat16Blob<S> & blob, int offset)
    {
        for(int j=0 ; j<W; j++)
        {
            int LE = 1; // not portable
            uint16_t * dst = reinterpret_cast<uint16_t*>(&data[j]);
            dst[LE] = blob.data[offset++];
        }
    }
    float & operator()(int i) { return data[i]; }
    float & operator[](int i) { return data[i]; }
    float const& operator()(int i) const { return data[i]; }
    float const& operator[](int i) const { return data[i]; }
};

float fastexp(float x)
{
    // TODO
    return std::exp(x);
}

struct RNG
{
    uint64_t x = 1234567890;

    float operator()()
    {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return (x % 100000) / float(100000);
    }
};

template<int W>
struct Sampler
{
    Vec<W> const& input;
    Vec<W> x;
    RNG rng;

    int operator()()
    {
        float sum_exp = 0;
        for(int i=0 ; i<W ; i++)
        {
            x[i] = (input[i] > -40) ? fastexp(input[i]) : 0;
            sum_exp += x[i];
        }
        float sample = rng() * sum_exp;
        float cumprob = 0;
        for(int i=0 ; i<W ; i++)
        {
            cumprob += x[i];
            if(cumprob > sample) { return i; }
        }
        return W-1;
    }
};

template<int H, int W>
struct Embed
{
    Mat<H, W> & weight;
    Vec<W> out;

    void operator()(int tok)
    {
        for(int i=0 ; i<W ; i++)
            out[i] = weight.data[tok][i];
    }
};

template<int W>
struct VecAdd
{
    Vec<W> & l;
    Vec<W> & r;
    Vec<W> out;

    void operator()()
    {
        for(int i=0 ; i<W ; i++)
            out[i] = l[i] + r[i];
    }
};



template<int W>
struct VecMul
{
    Vec<W> & l;
    Vec<W> & r;
    Vec<W> out;

    void operator()()
    {
        for(int i=0 ; i<W ; i++)
            out[i] = l[i] * r[i];
    }
};

template<int W>
struct VecMulAdd
{
    Vec<W> & x;
    Vec<W> & s;
    Vec<W> & b;
    Vec<W> out;

    void operator()()
    {
        for(int i=0 ; i<W ; i++)
            out[i] = x[i] * s[i] + b[i];
    }
};

template<int H, int W>
struct VecMat
{
    static constexpr int T = 16;
    static_assert(W % T == 0, "");

    Vec<H> const& x;
    Mat<H, W> const& m;
    Vec<W> out;

    void operator()()
    {
        for(int t = 0 ; t < W ; t += T)
        {
            for(int j = t ; j < t+T ; j++)
                out[j] = 0;
            for(int i = 0 ; i < H ; i++)
                for(int j = t ; j < t+T ; j++)
                    out[j] += x[i] * m.data[i][j];
        }
    }
};


template<int H, int W>
struct VecMatBias
{
    static constexpr int T = 16;
    static_assert(W % T == 0, "");

    Vec<H> const& x;
    Mat<H, W> const& m;
    Vec<W> const& b;
    Vec<W> out;

    void operator()()
    {
        for(int t = 0 ; t < W ; t += T)
        {
            for(int j = t ; j < t+T ; j++)
                out[j] = b[j];
            for(int i = 0 ; i < H ; i++)
                for(int j = t ; j < t+T ; j++)
                    out[j] += x[i] * m.data[i][j];
        }
    }
};


namespace ml {


template<int W>
void add(Vec<W> & out, Vec<W> const& l, Vec<W> const& r)
{
    for(int i=0 ; i<W ; i++)
        out[i] = l[i] * r[i];
}


} // namespace ml