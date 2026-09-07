// Offline direct-image sampling geometry. GPL-derived backend; see MODIFICATIONS.md.
#include "virtual_scene.hpp"
#include <algorithm>
#include <array>
#include <cfloat>
#include <atomic>
#include <future>
#include <numeric>
#include <stdexcept>

namespace sphorb {
static cv::Vec3f lower(cv::Vec3f a,cv::Vec3f b) {for(int k=0;k<3;++k)a[k]=std::min(a[k],b[k]);return a;}
static cv::Vec3f upper(cv::Vec3f a,cv::Vec3f b) {for(int k=0;k<3;++k)a[k]=std::max(a[k],b[k]);return a;}

VirtualScene::VirtualScene(const cv::Mat& vertices,const cv::Mat& triangles,const cv::Mat& uv,
    const cv::Mat& source_z,const cv::Mat& sources,float depth_band):depth_band_(depth_band) {
    if(vertices.type()!=CV_32F || vertices.cols!=3 || uv.type()!=CV_32F || uv.cols!=2 || uv.rows!=vertices.rows ||
       source_z.type()!=CV_32F || source_z.total()!=size_t(vertices.rows) || triangles.type()!=CV_32S || triangles.cols!=3 ||
       sources.type()!=CV_32S || sources.total()!=size_t(triangles.rows) ||
       !std::isfinite(depth_band) || depth_band<0 || depth_band>.2f) throw std::invalid_argument("Invalid virtual mesh arrays/band");
    for(int i=0;i<vertices.rows;++i) {
        for(int k=0;k<3;++k) if(!std::isfinite(vertices.at<float>(i,k))) throw std::invalid_argument("Nonfinite mesh vertex");
        for(int k=0;k<2;++k) if(!std::isfinite(uv.at<float>(i,k))) throw std::invalid_argument("Nonfinite source UV");
        if(!std::isfinite(source_z.ptr<float>()[i]) || source_z.ptr<float>()[i]<=0) throw std::invalid_argument("Invalid source depth");
    }
    for(int i=0;i<triangles.rows;++i) {
        std::array<cv::Vec3f,3> p;
        Triangle t;
        t.source=sources.ptr<int>()[i];
        if(t.source<0 || t.source>=64) throw std::invalid_argument("Source IDs must be 0..63");
        for(int k=0;k<3;++k) {
            int j=triangles.at<int>(i,k);
            if(j<0 || j>=vertices.rows) throw std::invalid_argument("Triangle vertex out of bounds");
            p[k]={vertices.at<float>(j,0),vertices.at<float>(j,1),vertices.at<float>(j,2)};
            t.z[k]=source_z.ptr<float>()[j]; t.uz[k]=uv.at<float>(j,0)*t.z[k]; t.vz[k]=uv.at<float>(j,1)*t.z[k];
        }
        t.a=p[0];t.e1=p[1]-p[0];t.e2=p[2]-p[0];
        t.lo=lower(lower(p[0],p[1]),p[2]);t.hi=upper(upper(p[0],p[1]),p[2]);
        triangles_.push_back(t); // Degenerate triangles are harmless and never intersect.
    }
    order_.resize(triangles_.size());std::iota(order_.begin(),order_.end(),0);
    nodes_.reserve(triangles_.size()/3+1);
    if(!triangles_.empty()) build(0,int(triangles_.size()));
}

int VirtualScene::build(int begin,int end) {
    Node n;n.begin=begin;n.end=end;
    n.lo=cv::Vec3f(FLT_MAX,FLT_MAX,FLT_MAX);n.hi=-n.lo;
    for(int i=begin;i<end;++i) {n.lo=lower(n.lo,triangles_[order_[i]].lo);n.hi=upper(n.hi,triangles_[order_[i]].hi);}
    int id=int(nodes_.size());nodes_.push_back(n);
    if(end-begin>8) {
        cv::Vec3f extent=n.hi-n.lo;int axis=extent[1]>extent[0]?1:0;if(extent[2]>extent[axis])axis=2;
        int mid=(begin+end)/2;
        std::nth_element(order_.begin()+begin,order_.begin()+mid,order_.begin()+end,[&](int a,int b){
            float x=triangles_[a].lo[axis]+triangles_[a].hi[axis],y=triangles_[b].lo[axis]+triangles_[b].hi[axis];
            return x==y?a<b:x<y;
        });
        int left=build(begin,mid),right=build(mid,end);
        nodes_[id].left=left;nodes_[id].right=right;
    }
    return id;
}

SurfaceHit VirtualScene::trace(cv::Vec3f ray,int requested_source) const {
    SurfaceHit result;
    if(nodes_.empty()) return result;
    std::array<float,64> depth;depth.fill(FLT_MAX);
    std::array<int,64> ids;ids.fill(-1);
    std::array<bool,64> front{};
    std::array<cv::Vec3f,64> bary;
    float nearest=FLT_MAX;
    // Median subdivision has depth < 32 for an int-indexed mesh.
    std::array<int,64> stack;int count=1;stack[0]=0;
    while(count) {
        const auto& node=nodes_[stack[--count]];
        float near=0,far=nearest==FLT_MAX?FLT_MAX:nearest*(1+depth_band_);
        bool box=true;
        for(int k=0;k<3;++k) {
            if(std::abs(ray[k])<1e-15f) {if(node.lo[k]>0 || node.hi[k]<0)box=false;}
            else {float a=node.lo[k]/ray[k],b=node.hi[k]/ray[k];near=std::max(near,std::min(a,b));far=std::min(far,std::max(a,b));}
        }
        if(!box || near>far)continue;
        if(node.left>=0) {stack[count++]=node.left;stack[count++]=node.right;continue;}
        for(int i=node.begin;i<node.end;++i) {
            int id=order_[i];const auto& t=triangles_[id];
            auto h=ray.cross(t.e2);float det=t.e1.dot(h);
            if(std::abs(det)<1e-12f)continue;
            float inv=1/det,u=(-t.a).dot(h)*inv;
            if(u<0 || u>1)continue;
            auto q=(-t.a).cross(t.e1);float v=ray.dot(q)*inv;
            if(v<0 || u+v>1)continue;
            float d=t.e2.dot(q)*inv;
            if(d<=1e-5f || !std::isfinite(d))continue;
            int s=t.source;
            if(d<depth[s] || (d==depth[s] && id<ids[s])) {depth[s]=d;ids[s]=id;bary[s]={1-u-v,u,v};front[s]=det>0;}
            nearest=std::min(nearest,d);
        }
    }
    // IDs are chronological window indices. Choose newest among near-front
    // surfaces, with per-source closest hits to preserve self-occlusion.
    for(int s=0;s<64;++s) if(ids[s]>=0 && front[s] && depth[s]<=nearest*(1+depth_band_)) result.visible_sources|=uint64_t(1)<<s;
    for(int s=63;s>=0;--s) if((requested_source<0 || requested_source==s) && ids[s]>=0 && front[s] && depth[s]<=nearest*(1+depth_band_)) {
        const auto& t=triangles_[ids[s]];auto b=bary[s];float z=b.dot(t.z);
        result.depth=depth[s];result.source=s;result.triangle=ids[s];
        result.uv={b.dot(t.uz)/z,b.dot(t.vz)/z};
        // Differentiate the ray/triangle intersection and perspective-correct
        // source UV to estimate a conservative isotropic source mip footprint.
        auto normal=t.e1.cross(t.e2);float nr=normal.dot(ray);
        cv::Vec3f axis=std::abs(ray[1])<.9f?cv::Vec3f(0,1,0):cv::Vec3f(1,0,0);
        auto dx=ray.cross(axis);dx/=float(cv::norm(dx));auto dy=ray.cross(dx);
        float g11=t.e1.dot(t.e1),g22=t.e2.dot(t.e2),g12=t.e1.dot(t.e2),den=g11*g22-g12*g12;
        float jac[4];int k=0;
        for(auto tangent:{dx,dy}) {
            auto dp=depth[s]*(tangent-ray*(normal.dot(tangent)/nr));
            float p1=dp.dot(t.e1),p2=dp.dot(t.e2);
            float du=(p1*g22-p2*g12)/den,dv=(p2*g11-p1*g12)/den;
            cv::Vec3f db(-du-dv,du,dv);
            jac[k++]=(db.dot(t.uz)-result.uv.x*db.dot(t.z))/z;
            jac[k++]=(db.dot(t.vz)-result.uv.y*db.dot(t.z))/z;
        }
        float a=jac[0]*jac[0]+jac[1]*jac[1],c=jac[2]*jac[2]+jac[3]*jac[3],cross=jac[0]*jac[2]+jac[1]*jac[3];
        result.pixel_scale=std::sqrt(std::max(0.f,.5f*(a+c+std::sqrt((a-c)*(a-c)+4*cross*cross))));
        if(!std::isfinite(result.pixel_scale))result.pixel_scale=FLT_MAX;
        break;
    }
    return result;
}

std::vector<SurfaceHit> VirtualScene::sample(const std::vector<cv::Vec3f>& rays,int workers,int requested_source) const {
    if(requested_source < -1 || requested_source>=64)throw std::invalid_argument("Invalid requested source");
    if(workers<1 || workers>64)throw std::invalid_argument("Workers must be 1..64");
    for(auto r:rays) if(!std::isfinite(r[0]) || !std::isfinite(r[1]) || !std::isfinite(r[2]) || std::abs(cv::norm(r)-1)>1e-4)
        throw std::invalid_argument("Expected finite unit rays");
    std::vector<SurfaceHit> out(rays.size());std::atomic<size_t> next{0};
    auto work=[&]{for(size_t start=next.fetch_add(256);start<rays.size();start=next.fetch_add(256))
        for(size_t i=start;i<std::min(start+256,rays.size());++i)out[i]=trace(rays[i],requested_source);};
    std::vector<std::future<void>> jobs;
    for(int i=0;i<workers;++i)jobs.emplace_back(std::async(std::launch::async,work));
    for(auto& job:jobs)job.get();return out;
}
}
