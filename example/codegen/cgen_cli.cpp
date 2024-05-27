#include <string>
#include <iostream>


extern "C"
int model_step(int prevtok, int numout, float temperature);


int main(int argc, char ** argv)
{
    std::string prompt;
    float temperature = 1;
    while(true)
    {
        std::getline(std::cin, prompt);

        for(uint8_t c : prompt) { model_step(c, 256, temperature); }

        uint8_t out = model_step('\n', 256, temperature);
        std::cout << out;

        for(int i=0 ; i<512 ; i++)
        {
            out = model_step(out, 256, temperature);
            std::cout << out;
            if(out == '\n') break;
        }
        
        std::cout << std::endl;
    }
}