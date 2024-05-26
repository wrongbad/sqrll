
#include <string>
#include <iostream>




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


int sample(float const* x, int n, RNG & rng)
{
    int out = 0;
    for(int i=0 ; i<n ; i++)
        if(x[i] > x[out]) out=i;
    return out;

    float sum_exp = 0;
    for(int i=0 ; i<n ; i++)
    {
        float e = (x[i] > -40) ? ml::fast_exp(x[i]) : 0;
        sum_exp += e;
    }
    float thresh = rng() * sum_exp;
    float cumprob = 0;
    for(int i=0 ; i<n ; i++)
    {
        float e = (x[i] > -40) ? ml::fast_exp(x[i]) : 0;
        cumprob += e;
        if(cumprob > thresh) { return i; }
    }
    return n-1;
}

int step(Model & model, int tok, RNG & rng, bool predict=true)
{
    model.x.ptr()[0] = tok;
    auto outs = model();
    model.mem = std::get<1>(outs);

    if(!predict) return 0;

    auto & logits = std::get<0>(outs);
    return sample(logits.ptr(), logits.numel(), rng);
}

int main(int argc, char ** argv)
{
    RNG rng;
    Model model;
    std::string prompt;
    while(true)
    {
        std::getline(std::cin, prompt);
        // std::cout << "you said: " << prompt << std::endl;

        for(uint8_t c : prompt) { step(model, c, rng, false); }

        uint8_t out = step(model, '\n', rng);

        std::cout << out;

        // for(int i=0 ; i<256 ; i++)
        // {
        //     std::cout << i << " " << model.sampler.x[i] << std::endl;
        // }

        for(int i=0 ; i<512 ; i++)
        {
            std::cout << (out = step(model, out, rng));
            if(out == '\n') break;
        }
        
        std::cout << std::endl;
    }
}