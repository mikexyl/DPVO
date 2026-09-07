// GPL-derived optional wrapper; see MODIFICATIONS.md.
#include "virtual_scene.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
namespace py=pybind11;
template<class T> using Array=py::array_t<T,py::array::c_style>;
void bind_virtual_scene(py::module_& m) {
    py::class_<sphorb::VirtualScene>(m,"VirtualScene")
      .def(py::init([](Array<float> vertices,Array<int> triangles,Array<float> uv,Array<float> z,Array<int> sources,float band) {
        if(vertices.ndim()!=2 || vertices.shape(1)!=3 || triangles.ndim()!=2 || triangles.shape(1)!=3 ||
           uv.ndim()!=2 || uv.shape(1)!=2 || uv.shape(0)!=vertices.shape(0) || z.ndim()!=1 || z.shape(0)!=vertices.shape(0) ||
           sources.ndim()!=1 || sources.shape(0)!=triangles.shape(0)) throw std::invalid_argument("Invalid virtual mesh dimensions");
        cv::Mat v(int(vertices.shape(0)),3,CV_32F,vertices.mutable_data()),t(int(triangles.shape(0)),3,CV_32S,triangles.mutable_data());
        cv::Mat u(int(uv.shape(0)),2,CV_32F,uv.mutable_data()),d(int(z.shape(0)),1,CV_32F,z.mutable_data());
        cv::Mat s(int(sources.shape(0)),1,CV_32S,sources.mutable_data());
        py::gil_scoped_release release;
        return std::make_unique<sphorb::VirtualScene>(v,t,u,d,s,band);
      }),py::arg("vertices").noconvert(),py::arg("triangles").noconvert(),py::arg("source_uv").noconvert(),
         py::arg("source_z").noconvert(),py::arg("sources").noconvert(),py::arg("depth_band")=.03f)
      .def("sample",[](const sphorb::VirtualScene& self,Array<float> rays,int workers,int requested_source) {
        if(rays.ndim()!=2 || rays.shape(1)!=3)throw std::invalid_argument("Expected Nx3 float32 rays");
        std::vector<cv::Vec3f> input(size_t(rays.shape(0)));
        std::memcpy(input.data(),rays.data(),input.size()*sizeof(cv::Vec3f));
        std::vector<sphorb::SurfaceHit> hits;
        {py::gil_scoped_release release;hits=self.sample(input,workers,requested_source);}
        py::ssize_t n=hits.size();py::array_t<float> uv({n,py::ssize_t(2)}),depth(n),scale(n);
        py::array_t<int> sources(n),triangles(n);py::array_t<uint64_t> visible(n);
        for(py::ssize_t i=0;i<n;++i) {
            uv.mutable_at(i,0)=hits[i].uv.x;uv.mutable_at(i,1)=hits[i].uv.y;
            depth.mutable_at(i)=hits[i].depth;scale.mutable_at(i)=hits[i].pixel_scale;
            visible.mutable_at(i)=hits[i].visible_sources;
            sources.mutable_at(i)=hits[i].source;triangles.mutable_at(i)=hits[i].triangle;
        }
        py::dict out;out["uv"]=uv;out["depth"]=depth;out["pixel_scale"]=scale;out["source"]=sources;out["triangle"]=triangles;out["visible_sources"]=visible;return out;
      },py::arg("rays").noconvert(),py::arg("workers")=4,py::arg("requested_source")=-1);
}
