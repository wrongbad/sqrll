
#include <string>
#include <iostream>

int main(int argc, char ** argv)
{
    std::string prompt;
    while(true)
    {
        std::getline(std::cin, prompt);
        std::cout << "you said: " << prompt << std::endl;

        char out = '\n';
        for(char c : prompt) { out = model.step(c); }
        std::cout << out;

        // for(int i=0 ; i<256 ; i++)
        // {
        //     std::cout << i << " " << model.sampler.x[i] << std::endl;
        // }

        for(int i=0 ; i<512 ; i++)
        {
            std::cout << (out = model.step(out));
            if(out == '\n') break;
        }
        
        std::cout << std::endl;
    }
}