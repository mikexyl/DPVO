// GPL-derived SPHORB backend; see upstream/README.md and MODIFICATIONS.md.
#include "core.hpp"
#include "kernels.hpp"
#include <opencv2/imgcodecs.hpp>
#include <openssl/sha.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace sphorb {
namespace fs = std::filesystem;
constexpr int cells[] = {256, 204, 162, 128, 102, 80, 64};
constexpr int edge = 18;
#include "tables_hashes.hpp"
struct LevelTables {
    std::vector<float> geo;
    std::array<std::vector<float>, 5> sampling;
    cv::Mat mask;
};
struct Tables { std::array<LevelTables, 7> levels; };

static std::vector<float> read_table(const fs::path& path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    std::string magic;
    int w=0, h=0;
    float scale=0;
    f >> magic >> w >> h >> scale;
    if (!f || magic != "PF" || w <= 0 || h <= 0 || size_t(w)*h*3 != expected ||
        (scale != 1 && scale != -1)) throw std::runtime_error("Invalid PFM header: " + path.string());
    char delimiter;
    f.get(delimiter);
    if (delimiter != '\r' && delimiter != '\n') throw std::runtime_error("Invalid PFM delimiter");
    std::vector<float> data(expected), row(size_t(w)*3);
    f.read(reinterpret_cast<char*>(data.data()), expected*sizeof(float));
    if (!f || f.peek() != EOF) throw std::runtime_error("Truncated or oversized table: " + path.string());
    // Upstream uses native little-endian floats even for positive scale (top-down).
    if (scale < 0) for (int y=0; y<h/2; ++y)
        std::swap_ranges(data.begin()+size_t(y)*w*3, data.begin()+size_t(y+1)*w*3,
                         data.begin()+size_t(h-1-y)*w*3);
    for (float v : data) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite table value");
    return data;
}

static std::shared_ptr<const Tables> load_tables(const std::string& directory) {
    fs::path path(directory);
    if (!path.is_absolute()) throw std::invalid_argument("Table directory must be absolute");
    path = fs::canonical(path);
    // Validate even when an immutable cached copy exists, so corrupt/missing assets never go unnoticed.
    for (const auto& [name, expected] : table_hashes) {
        std::ifstream file(path/name, std::ios::binary);
        if (!file) throw std::runtime_error("Missing SPHORB table: " + (path/name).string());
        std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
        unsigned char digest[SHA256_DIGEST_LENGTH];
        SHA256(bytes.data(), bytes.size(), digest);
        std::ostringstream hex;
        for (auto c : digest) hex << std::hex << std::setfill('0') << std::setw(2) << int(c);
        if (hex.str() != expected) throw std::runtime_error("SPHORB table checksum mismatch: " + name);
    }
    static std::mutex cache_mutex;
    static std::map<std::string, std::weak_ptr<const Tables>> cache;
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = cache.find(path.string());
    if (found != cache.end()) if (auto existing=found->second.lock()) return existing;
    auto tables = std::make_shared<Tables>();
    for (int l=0; l<7; ++l) {
        int n=cells[l], count=(n+1)*(2*n+1);
        auto& t=tables->levels[l];
        t.geo=read_table(path/("geoinfo"+std::to_string(n)+".pfm"), count*3);
        for (int i=0; i<5; ++i) {
            auto& v=t.sampling[i];
            v=read_table(path/("imginfo"+std::to_string(n)+"_"+std::to_string(i)+".pfm"), ((count*4+2)/3)*3);
            for (int k=0; k<count; ++k) {
                if (v[k*4]<0 || v[k*4]>=n*5 || v[k*4+1]<0 || v[k*4+1]>=n*5/2 ||
                    v[k*4+2]<0 || v[k*4+2]>1 || v[k*4+3]<0 || v[k*4+3]>1)
                    throw std::runtime_error("Out-of-range sampling table");
            }
        }
        for (int k=0; k<count; ++k) {
            float norm=cv::norm(cv::Vec3f(t.geo[k*3], t.geo[k*3+1], t.geo[k*3+2]));
            if (std::abs(norm-1)>1e-4) throw std::runtime_error("Invalid geodesic bearing");
        }
        t.mask=cv::imread((path/("mask"+std::to_string(n)+".bmp")).string(), cv::IMREAD_GRAYSCALE);
        if (t.mask.size()!=cv::Size(2*n+36,n+36)) throw std::runtime_error("Invalid grid mask dimensions");
    }
    for (auto it=cache.begin(); it!=cache.end();) {
        if (it->second.expired()) it=cache.erase(it); else ++it;
    }
    cache[path.string()]=tables;
    return tables;
}

static void extend(const std::array<cv::Mat,5>& raw, std::array<cv::Mat,5>& extended) {
    for (int i=0; i<5; ++i) {
        auto& dst=extended[i];
        dst.create(raw[i].rows+2*edge-1, raw[i].cols+2*edge-1, CV_8U);
        dst.setTo(0);
        raw[i].copyTo(dst(cv::Rect(edge-1, edge, raw[i].cols, raw[i].rows)));
        extendTopRight(dst, raw[(i+1)%5], edge);
        extendBottomLeft(dst, raw[(i+4)%5], edge);
    }
}

static bool hex_support(const cv::Mat& mask, int x, int y, int radius, int expected=-1) {
    if (x-radius<0 || y-radius<0 || x+radius>=mask.cols || y+radius>=mask.rows) return false;
    for (int dy=-radius; dy<=radius; ++dy)
        for (int dx=std::max(-radius,-dy-radius); dx<=std::min(radius,radius-dy); ++dx)
            if (expected<0 ? !mask.at<uchar>(y+dy,x+dx) : mask.at<uchar>(y+dy,x+dx)!=expected) return false;
    return true;
}

static bool descriptor_support(const cv::Mat& mask, const cv::KeyPoint& kp, const cv::Mat& sources=cv::Mat()) {
    float angle=kp.angle*float(CV_PI/180.f);
    float a=std::cos(angle), b=std::sin(angle), c=std::sqrt(3.f), d=b*c/3;
    b=a-d; a=a+d; c=2*d;
    for (int k=0; k<512; ++k) {
        int x=cvRound(kp.pt.x)+cvRound(bit_pattern[k*2]*b-bit_pattern[k*2+1]*c);
        int y=cvRound(kp.pt.y)+cvRound(bit_pattern[k*2+1]*a+bit_pattern[k*2]*c);
        if (x<0 || y<0 || x>=mask.cols || y>=mask.rows || !mask.at<uchar>(y,x)) return false;
        if (!sources.empty() && sources.at<uchar>(y,x)!=sources.at<uchar>(cvRound(kp.pt.y),cvRound(kp.pt.x))) return false;
    }
    return true;
}

Extractor::Extractor(const std::string& directory, int features, int levels, int threshold, int workers)
    : features_(features), levels_(levels), threshold_(threshold), workers_(workers) {
    auto start=std::chrono::steady_clock::now();
    if (features<1 || levels<1 || levels>7 || threshold<1 || threshold>254 || workers<1 || workers>64)
        throw std::invalid_argument("Need features>0, levels 1..7, threshold 1..254, workers 1..64");
    tables_=load_tables(directory);
    static std::once_flag threading;
    std::call_once(threading,[]{ cv::setNumThreads(1); });
    buffers_.resize(levels);
    quotas_.resize(levels);
    float factor=float(1.0/std::pow(2.,1/3.0));
    float desired=features*(1-factor)/(1-float(std::pow(double(factor),double(levels))));
    int sum=0;
    for (int l=0; l<levels-1; ++l) { quotas_[l]=std::min(cvRound(desired), features-sum); sum+=quotas_[l]; desired*=factor; }
    quotas_.back()=features-sum;
    setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}

std::vector<Feature> Extractor::level(const cv::Mat& gray, const cv::Mat& valid, int l, bool legacy) {
    const auto& t=tables_->levels[l];
    auto& b=buffers_[l];
    int n=cells[l];
    cv::Size size(n*5,n*5/2);
    cv::resize(gray,b.image,size,0,0,cv::INTER_AREA);
    // Every nonzero contribution of an unknown source pixel invalidates the resized pixel.
    cv::compare(valid,0,b.invalid,cv::CMP_EQ);
    b.invalid.convertTo(b.invalid,CV_32F,1./255);
    cv::resize(b.invalid,b.resized_invalid,size,0,0,cv::INTER_AREA);
    for (int i=0; i<5; ++i) {
        b.raw[i].create(n+1,2*n+1,CV_8U);
        b.valid[i].create(n+1,2*n+1,CV_8U);
        const float* info=t.sampling[i].data();
        for (int y=0; y<=n; ++y) for (int x=0; x<=2*n; ++x,info+=4) {
            float wh=info[2], wv=info[3];
            int ix=int(info[0]), iy=int(info[1]);
            if (!legacy) {
                float u=info[0]+(1-wh)-.5f, v=info[1]+(1-wv)-.5f;
                ix=int(std::floor(u)); iy=int(std::floor(v));
                wh=1-(u-ix); wv=1-(v-iy);
            }
            int x0=(ix%size.width+size.width)%size.width, x1=(x0+1)%size.width;
            int y0=std::clamp(iy,0,size.height-1), y1=std::clamp(iy+1,0,size.height-1);
            float v12=b.image.at<uchar>(y0,x0)*wh+b.image.at<uchar>(y0,x1)*(1-wh);
            float v34=b.image.at<uchar>(y1,x0)*wh+b.image.at<uchar>(y1,x1)*(1-wh);
            b.raw[i].at<uchar>(y,x)=uchar(v12*wv+v34*(1-wv));
            bool ok=true;
            for (auto [sx,wx] : {std::pair{x0,wh},std::pair{x1,1-wh}})
                for (auto [sy,wy] : {std::pair{y0,wv},std::pair{y1,1-wv}})
                    if (wx*wy>0 && b.resized_invalid.at<float>(sy,sx)>0) ok=false;
            b.valid[i].at<uchar>(y,x)=ok?255:0;
        }
    }
    return detect_level(l,gray.size(),false);
}

std::vector<Feature> Extractor::detect_level(int l, cv::Size display_size, bool coherent) {
    const auto& t=tables_->levels[l];
    auto& b=buffers_[l];
    int n=cells[l];
    extend(b.raw,b.extended);
    extend(b.valid,b.support);
    if (coherent) extend(b.sources,b.extended_sources);
    cv::Mat kernel_mat(7,7,CV_64F,kernel), footprint=kernel_mat!=0;
    std::vector<cv::KeyPoint> keypoints;
    for (int i=0; i<5; ++i) {
        cv::erode(b.support[i],b.detect_mask[i],footprint,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
        cv::Mat same_source;
        if (coherent) {
            cv::Mat minimum,maximum;
            cv::erode(b.extended_sources[i],minimum,footprint,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
            cv::dilate(b.extended_sources[i],maximum,footprint,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
            same_source=(minimum==maximum)&(minimum!=0);
            cv::bitwise_and(b.detect_mask[i],same_source,b.detect_mask[i]);
        }
        cv::bitwise_and(b.detect_mask[i],t.mask,b.detect_mask[i]);
        int count=0;
        using Corners=std::unique_ptr<xy,decltype(&std::free)>;
        using Scores=std::unique_ptr<int,decltype(&std::free)>;
        Corners corners(sfast_corner_detect(b.extended[i].ptr(),b.detect_mask[i].ptr(),
                        b.extended[i].cols,int(b.extended[i].step),b.extended[i].rows,threshold_,&count),std::free);
        Scores scores(sfastScore(b.extended[i].ptr(),int(b.extended[i].step),corners.get(),count,threshold_),std::free);
        std::vector<cv::KeyPoint> part;
        sfastNonmaxSuppression(corners.get(),scores.get(),count,part,i);
        cv::filter2D(b.extended[i],b.smooth[i],-1,kernel_mat);
        cv::erode(b.support[i],b.smooth_support[i],footprint,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
        if (coherent) cv::bitwise_and(b.smooth_support[i],same_source,b.smooth_support[i]);
        for (auto kp:part) {
            int x=cvRound(kp.pt.x), y=cvRound(kp.pt.y);
            if (x-edge+1<0 || x-edge+1>2*n || y-edge<0 || y-edge>n || !hex_support(b.support[i],x,y,15)) continue;
            if (coherent && !hex_support(b.extended_sources[i],x,y,15,b.extended_sources[i].at<uchar>(y,x))) continue;
            kp.angle=IC_Angle(b.extended[i],15,kp.pt,t.geo.data());
            if (descriptor_support(b.smooth_support[i],kp,coherent?b.extended_sources[i]:cv::Mat())) keypoints.push_back(kp);
        }
    }
    std::stable_sort(keypoints.begin(),keypoints.end(),[](const auto& a,const auto& b){return a.response>b.response;});
    if (keypoints.size()>size_t(quotas_[l])) keypoints.resize(quotas_[l]);
    std::array<cv::Point,512> pattern;
    for (int k=0; k<512; ++k) pattern[k]={bit_pattern[k*2],bit_pattern[k*2+1]};
    std::vector<Feature> out;
    out.reserve(keypoints.size());
    for (const auto& kp:keypoints) {
        int x=cvRound(kp.pt.x)-edge+1, y=cvRound(kp.pt.y)-edge;
        const float* g=&t.geo[(x+y*(2*n+1))*3];
        double angle=kp.class_id*2*CV_PI/5, co=std::cos(angle), si=std::sin(angle);
        double gx=co*g[0]-si*g[1], gy=co*g[1]+si*g[0], gz=std::clamp(double(g[2]),-1.,1.);
        // Upstream longitude=atan2(gy,gx)+pi, colatitude=acos(gz).
        // DPVO longitude=atan2(R,F), latitude=asin(D), continuous pixel centers.
        cv::Vec3f ray(gy,-gz,gx); ray/=cv::norm(ray);
        Feature f;
        f.bearing=ray;
        f.uv={float((std::atan2(double(ray[0]),double(ray[2]))/(2*CV_PI)+.5)*display_size.width-.5),
              float((std::asin(std::clamp(double(ray[1]),-1.,1.))/CV_PI+.5)*display_size.height-.5)};
        if (f.uv.x>=display_size.width-.5f) f.uv.x=-.5f;
        f.response=kp.response; f.angle=kp.angle; f.size=31.f*float(display_size.width)/(5*n);
        f.octave=l; f.section=kp.class_id; f.grid=kp.pt;
        computeOrbDescriptor(kp,b.smooth[kp.class_id],pattern.data(),f.descriptor.data(),32);
        out.push_back(f);
    }
    return out;
}

std::vector<cv::Vec3f> Extractor::grid_bearings(int l) const {
    if (l<0 || l>=levels_) throw std::invalid_argument("Invalid grid level");
    const auto& geo=tables_->levels[l].geo;
    std::vector<cv::Vec3f> rays;
    rays.reserve(5*geo.size()/3);
    for (int i=0;i<5;++i) {
        double co=std::cos(i*2*CV_PI/5), si=std::sin(i*2*CV_PI/5);
        for(size_t k=0;k<geo.size();k+=3) {
            double gx=co*geo[k]-si*geo[k+1],gy=co*geo[k+1]+si*geo[k];
            cv::Vec3f ray(gy,-std::clamp(double(geo[k+2]),-1.,1.),gx);
            rays.push_back(ray/float(cv::norm(ray)));
        }
    }
    return rays;
}

std::vector<Feature> Extractor::extract_grids(const std::vector<std::array<cv::Mat,5>>& gray,
        const std::vector<std::array<cv::Mat,5>>& valid,
        const std::vector<std::array<cv::Mat,5>>& sources, cv::Size display_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (int(gray.size())!=levels_ || valid.size()!=gray.size() || sources.size()!=gray.size() ||
        display_size.width<2 || display_size.height<2) throw std::invalid_argument("Invalid direct grid inputs");
    for(int l=0;l<levels_;++l) for(int i=0;i<5;++i) {
        auto expected=cv::Size(2*cells[l]+1,cells[l]+1);
        for(const auto* m:{&gray[l][i],&valid[l][i],&sources[l][i]})
            if(m->type()!=CV_8U || m->size()!=expected) throw std::invalid_argument("Invalid direct grid shape/type");
        if(cv::countNonZero((valid[l][i]!=0)&(sources[l][i]==0))) throw std::invalid_argument("Observed sample needs a source ID");
        gray[l][i].copyTo(buffers_[l].raw[i]); valid[l][i].copyTo(buffers_[l].valid[i]);
        sources[l][i].copyTo(buffers_[l].sources[i]);
        buffers_[l].raw[i].setTo(0,valid[l][i]==0);
        buffers_[l].sources[i].setTo(0,valid[l][i]==0);
    }
    std::vector<std::vector<Feature>> results(levels_);
    std::atomic<int> next{0};
    auto work=[&] {for(int l=next.fetch_add(1);l<levels_;l=next.fetch_add(1)) results[l]=detect_level(l,display_size,true);};
    std::vector<std::future<void>> workers;
    for(int i=0;i<std::min(workers_,levels_);++i) workers.emplace_back(std::async(std::launch::async,work));
    for(auto& worker:workers) worker.get();
    std::vector<Feature> output;
    for(auto& result:results) output.insert(output.end(),result.begin(),result.end());
    return output;
}

std::vector<Feature> Extractor::extract(const cv::Mat& gray, const cv::Mat& valid, bool legacy) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gray.type()!=CV_8UC1 || valid.type()!=CV_8UC1 || gray.size()!=valid.size() || gray.rows<2 || gray.cols<2)
        throw std::invalid_argument("Expected matching uint8 gray and validity images of at least 2x2");
    if (!cv::countNonZero(valid)) return {};
    cv::Mat clean=gray.clone(); clean.setTo(0,valid==0);
    std::vector<std::vector<Feature>> results(levels_);
    std::atomic<int> next{0};
    auto work=[&] { for (int l=next.fetch_add(1); l<levels_; l=next.fetch_add(1)) results[l]=level(clean,valid,l,legacy); };
    std::vector<std::future<void>> workers;
    for (int i=0; i<std::min(workers_,levels_); ++i) workers.emplace_back(std::async(std::launch::async,work));
    for (auto& worker:workers) worker.get();
    std::vector<Feature> output;
    for (auto& result:results) output.insert(output.end(),result.begin(),result.end());
    return output;
}
}
