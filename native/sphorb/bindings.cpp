// GPL-derived optional wrapper; see MODIFICATIONS.md.
#include "core.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <cstring>
namespace py=pybind11;
void bind_virtual_scene(py::module_&);
static py::dict pack(const std::vector<sphorb::Feature>& features,double seconds,bool addresses) {
py::ssize_t n=features.size();
py::array_t<uint8_t> desc({n,py::ssize_t(32)});
py::array_t<float> uv({n,py::ssize_t(2)}), bearings({n,py::ssize_t(3)}), grid({n,py::ssize_t(2)});
py::array_t<float> response(n),angles(n),sizes(n);
py::array_t<int> octaves(n),sections(n);
for (py::ssize_t i=0;i<n;++i) {
    const auto& f=features[i];
    std::memcpy(desc.mutable_data(i,0),f.descriptor.data(),32);
    std::memcpy(bearings.mutable_data(i,0),f.bearing.val,3*sizeof(float));
    uv.mutable_at(i,0)=f.uv.x; uv.mutable_at(i,1)=f.uv.y;
    grid.mutable_at(i,0)=f.grid.x; grid.mutable_at(i,1)=f.grid.y;
    response.mutable_at(i)=f.response; angles.mutable_at(i)=f.angle; sizes.mutable_at(i)=f.size;
    octaves.mutable_at(i)=f.octave; sections.mutable_at(i)=f.section;
}
py::dict result;
result["descriptors"]=desc; result["uv"]=uv; result["bearings"]=bearings;
result["responses"]=response; result["orientations"]=angles; result["sizes"]=sizes;
result["octaves"]=octaves; result["native_seconds"]=seconds;
// Grid addresses exist only for native/reference audits, never cube-face metadata.
if (addresses) {result["grid"]=grid; result["sections"]=sections;}
return result;
}
PYBIND11_MODULE(_sphorb, m) {
    bind_virtual_scene(m);
    m.attr("descriptor_family")="sphorb";
    m.attr("descriptor_version")="e5f2ccf-mask-rdf-v1";
    m.attr("opencv_version")=CV_VERSION;
    py::class_<sphorb::Extractor>(m,"Extractor")
        .def(py::init<const std::string&,int,int,int,int>(),py::arg("tables"),py::arg("features")=3000,
             py::arg("levels")=7,py::arg("threshold")=20,py::arg("workers")=4,
             py::call_guard<py::gil_scoped_release>())
        .def_readonly("setup_seconds",&sphorb::Extractor::setup_seconds)
        .def("quotas",[](const sphorb::Extractor& self){py::list out;for(int q:self.quotas())out.append(q);return out;})
        .def("extract",[](sphorb::Extractor& self,py::array_t<uint8_t,py::array::c_style> gray,
                          py::array_t<uint8_t,py::array::c_style> valid,bool legacy) {
            if (gray.ndim()!=2 || valid.ndim()!=2 || gray.shape(0)!=valid.shape(0) || gray.shape(1)!=valid.shape(1))
                throw std::invalid_argument("Expected matching contiguous uint8 2D gray/valid images");
            cv::Mat g(int(gray.shape(0)),int(gray.shape(1)),CV_8U,gray.mutable_data());
            cv::Mat v(int(valid.shape(0)),int(valid.shape(1)),CV_8U,valid.mutable_data());
            std::vector<sphorb::Feature> features;
            auto start=std::chrono::steady_clock::now();
            { py::gil_scoped_release release; features=self.extract(g,v,legacy); }
            double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            return pack(features,seconds,legacy);
        },py::arg("gray").noconvert(),py::arg("valid").noconvert(),py::arg("upstream_sampling")=false)
        .def("grid_bearings",[](const sphorb::Extractor& self,int level) {
            auto rays=self.grid_bearings(level);
            static const int cells[]={256,204,162,128,102,80,64};
            py::array_t<float> out({py::ssize_t(5),py::ssize_t(cells[level]+1),py::ssize_t(2*cells[level]+1),py::ssize_t(3)});
            std::memcpy(out.mutable_data(),rays.data(),rays.size()*sizeof(cv::Vec3f));return out;
        })
        .def("extract_grids",[](sphorb::Extractor& self,py::list images,py::list masks,py::list sources,int width,int height) {
            auto read=[](py::list arrays) {
                std::vector<std::array<cv::Mat,5>> result;
                for(auto item:arrays) {
                    if(!py::isinstance<py::array_t<uint8_t>>(item) ||
                       !(py::reinterpret_borrow<py::array>(item).flags()&py::array::c_style))
                        throw std::invalid_argument("Expected contiguous uint8 grids without conversion");
                    auto a=py::cast<py::array_t<uint8_t,py::array::c_style>>(item);
                    if(a.ndim()!=3 || a.shape(0)!=5)throw std::invalid_argument("Expected contiguous uint8 [5,H,W] grids");
                    std::array<cv::Mat,5> parts;
                    for(int i=0;i<5;++i)parts[i]=cv::Mat(int(a.shape(1)),int(a.shape(2)),CV_8U,a.mutable_data(i,0,0));
                    result.push_back(parts);
                }
                return result;
            };
            auto gray=read(images),valid=read(masks),ids=read(sources);
            std::vector<sphorb::Feature> features;
            auto start=std::chrono::steady_clock::now();
            {py::gil_scoped_release release;features=self.extract_grids(gray,valid,ids,cv::Size(width,height));}
            return pack(features,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),true);
        },py::arg("gray"),py::arg("valid"),py::arg("sources"),py::arg("width")=1024,py::arg("height")=512);
}
