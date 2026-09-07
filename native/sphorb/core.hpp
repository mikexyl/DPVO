// GPL-derived SPHORB backend; see upstream/README.md and MODIFICATIONS.md.
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sphorb {
struct Tables;
struct Feature {
    std::array<unsigned char, 32> descriptor{};
    cv::Vec3f bearing;
    cv::Point2f uv, grid;
    float response, angle, size;
    int octave, section;
};
struct Buffers {
    cv::Mat image, invalid, resized_invalid;
    std::array<cv::Mat, 5> raw, valid, extended, support, smooth, smooth_support, detect_mask;
    std::array<cv::Mat, 5> sources, extended_sources;
};
class Extractor {
public:
    Extractor(const std::string& directory, int features=3000, int levels=7, int threshold=20, int workers=4);
    std::vector<Feature> extract(const cv::Mat& gray, const cv::Mat& valid, bool upstream_sampling=false);
    std::vector<cv::Vec3f> grid_bearings(int level) const;
    std::vector<Feature> extract_grids(const std::vector<std::array<cv::Mat,5>>& gray,
        const std::vector<std::array<cv::Mat,5>>& valid,
        const std::vector<std::array<cv::Mat,5>>& sources, cv::Size display_size);
    const std::vector<int>& quotas() const {return quotas_;}
    double setup_seconds=0;
private:
    std::shared_ptr<const Tables> tables_;
    int features_, levels_, threshold_, workers_;
    std::vector<int> quotas_;
    std::vector<Buffers> buffers_;
    std::mutex mutex_;
    std::vector<Feature> level(const cv::Mat&, const cv::Mat&, int, bool);
    std::vector<Feature> detect_level(int, cv::Size, bool coherent);
};
}
