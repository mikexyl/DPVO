// Native memory and published-algorithm regression audit (GPL-derived).
#include "core.hpp"
#include "virtual_scene.hpp"
#include "reference/reference.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <map>
#include <tuple>

static void require(bool condition,const char* why) { if(!condition) throw std::runtime_error(why); }
int main(int argc,char** argv) {
    if(argc!=2) { std::cerr<<"usage: sphorb_audit /absolute/table/directory\n"; return 2; }
    try {
        cv::Mat gray(640,1280,CV_8U), mask(gray.size(),CV_8U,cv::Scalar(255));
        cv::RNG rng(42); rng.fill(gray,cv::RNG::UNIFORM,0,256);
        // Large quota compares all accepted full-support sites, with identical legacy table sampling.
        sphorb::Extractor native(argv[1],1000000,7,20,4);
        auto actual=native.extract(gray,mask,true);
        cv::SPHORB reference(argv[1],1000000,7,20);
        std::vector<cv::KeyPoint> kps; cv::Mat descriptors;
        reference(gray,mask,kps,descriptors);
        using Key=std::tuple<int,int,int,int>;
        std::map<Key,int> lookup;
        for(size_t i=0;i<kps.size();++i) lookup[{kps[i].octave,kps[i].class_id,cvRound(kps[i].pt.x),cvRound(kps[i].pt.y)}]=int(i);
        size_t matched=0, absent=0;
        for(const auto& f:actual) {
            auto it=lookup.find({f.octave,f.section,cvRound(f.grid.x),cvRound(f.grid.y)});
            if(it==lookup.end()) {++absent;continue;}
            int row=it->second;
            require(std::abs(f.angle-kps[row].angle)<1e-5,"orientation differs from upstream");
            require(f.response==kps[row].response,"FAST score differs from upstream");
            require(std::equal(f.descriptor.begin(),f.descriptor.end(),descriptors.ptr<uchar>(row)),"descriptor differs from upstream");
            ++matched;
        }
        require(matched>1000 && absent==0,"insufficient upstream agreement");
        std::cout<<"{\"reference_features\":"<<kps.size()<<",\"supported_features\":"<<actual.size()
                 <<",\"identical_descriptors\":"<<matched<<",\"new_sites\":"<<absent<<"}"<<std::endl;
        sphorb::Extractor one(argv[1],3000,7,20,1), four(argv[1],3000,7,20,4);
        for(int pass=0;pass<4;++pass) {
            if(pass==1) mask(cv::Rect(0,0,240,220)).setTo(0);
            if(pass==2) mask(cv::Rect(550,100,200,430)).setTo(0);
            if(pass==3) mask.setTo(0);
            auto a=one.extract(gray,mask),b=four.extract(gray,mask);
            require(a.size()==b.size(),"thread count changed feature count");
            for(size_t i=0;i<a.size();++i) require(a[i].descriptor==b[i].descriptor && a[i].uv==b[i].uv,"thread count changed output");
        }
        for(cv::Size size:{cv::Size(2,2),cv::Size(101,63),cv::Size(2048,1024)}) {
            cv::resize(gray,gray,size); cv::Mat support(size,CV_8U,cv::Scalar(255));
            four.extract(gray,support);
        }
        std::cout<<"native lifetime, bounds, empty mask, and threading audit passed"<<std::endl;
        float vertex_data[]={-1,-1,2,1,-1,2,-1,1,2,1,1,2};
        float uv_data[]={0,0,100,0,0,100,100,100},z_data[]={2,2,2,2};
        int tri_data[]={0,2,1,1,2,3},source_data[]={0,0};
        cv::Mat vertices(4,3,CV_32F,vertex_data),triangles(2,3,CV_32S,tri_data),uv(4,2,CV_32F,uv_data);
        cv::Mat z(4,1,CV_32F,z_data),sources(2,1,CV_32S,source_data);
        sphorb::VirtualScene scene(vertices,triangles,uv,z,sources);
        auto hits=scene.sample({cv::Vec3f(0,0,1),cv::Vec3f(0,0,-1)},4);
        require(hits[0].source==0 && std::abs(hits[0].depth-2)<1e-6 && cv::norm(hits[0].uv-cv::Point2f(50,50))<1e-4,"virtual ray/UV regression");
        require(hits[1].source==-1 && hits[0].visible_sources==1,"virtual occlusion regression");
        std::vector<std::array<cv::Mat,5>> grids(7),valid(7),ids(7);
        const int cells[]={256,204,162,128,102,80,64};
        for(int l=0;l<7;++l)for(int i=0;i<5;++i) {
            grids[l][i]=cv::Mat(cells[l]+1,2*cells[l]+1,CV_8U);rng.fill(grids[l][i],cv::RNG::UNIFORM,0,256);
            valid[l][i]=cv::Mat(grids[l][i].size(),CV_8U,cv::Scalar(255));ids[l][i]=cv::Mat(grids[l][i].size(),CV_8U,cv::Scalar(1));
        }
        auto direct=one.extract_grids(grids,valid,ids,cv::Size(1024,512));
        require(direct.size()>100,"virtual direct grid detection regression");
        for(auto& level:valid)for(auto& mask:level)mask.setTo(0);
        require(one.extract_grids(grids,valid,ids,cv::Size(1024,512)).empty(),"virtual empty support regression");
        std::cout<<"virtual scene and direct-grid memory audit passed"<<std::endl;
    } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; }
}
