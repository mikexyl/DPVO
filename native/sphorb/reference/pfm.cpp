#include "pfm.h"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
bool read_pfm(const char* name,float* output,std::size_t expected) {
    std::ifstream f(name,std::ios::binary); std::string magic; int w=0,h=0; float scale=0;
    f>>magic>>w>>h>>scale; if(!f || magic!="PF" || w<1 || h<1 || w>200000 || h>1000) return false;
    if (size_t(w)*h*3 != expected || (scale!=1 && scale!=-1)) return false;
    char separator; f.get(separator); if(separator!='\r' && separator!='\n') return false;
    f.read(reinterpret_cast<char*>(output),size_t(w)*h*3*sizeof(float));
    if (!f || f.peek()!=EOF) return false;
    if(scale<0) for(int y=0;y<h/2;++y) std::swap_ranges(output+size_t(y)*w*3,output+size_t(y+1)*w*3,output+size_t(h-1-y)*w*3);
    return true;
}
