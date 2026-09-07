// Offline direct-image sampling geometry. GPL-derived backend; see MODIFICATIONS.md.
#pragma once
#include <opencv2/core.hpp>
#include <cmath>
#include <limits>
#include <vector>

namespace sphorb {
struct SurfaceHit {
    float depth=std::numeric_limits<float>::quiet_NaN();
    cv::Point2f uv{NAN,NAN};
    float pixel_scale=0;
    int source=-1, triangle=-1;
    uint64_t visible_sources=0;
};
class VirtualScene {
public:
    VirtualScene(const cv::Mat& vertices,const cv::Mat& triangles,const cv::Mat& uv,
                 const cv::Mat& source_z,const cv::Mat& sources,float depth_band=.03f);
    std::vector<SurfaceHit> sample(const std::vector<cv::Vec3f>& rays,int workers=4,int requested_source=-1) const;
private:
    struct Triangle {cv::Vec3f a,e1,e2,lo,hi,z; cv::Vec3f uz,vz; int source;};
    struct Node {cv::Vec3f lo,hi; int left=-1,right=-1,begin=0,end=0;};
    std::vector<Triangle> triangles_;
    std::vector<int> order_;
    std::vector<Node> nodes_;
    float depth_band_;
    int build(int,int);
    SurfaceHit trace(cv::Vec3f,int requested_source) const;
};
}
