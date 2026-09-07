// Optional bindings for upstream MIT-SPARK TEASER++; see THIRD_PARTY.md.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <teaser/registration.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <limits>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;
using Cloud = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct ThreadLimit {
  int previous = omp_get_max_threads();
  explicit ThreadLimit(int n) { omp_set_num_threads(n); }
  ~ThreadLimit() { omp_set_num_threads(previous); }
};

py::dict solve(const Cloud& source, const Cloud& target, double bound, int workers, double limit) {
  if (source.rows() != target.rows() || source.rows() < 3 || source.rows() > 4096 ||
      !source.allFinite() || !target.allFinite() || !std::isfinite(bound) || bound <= 0 ||
      workers < 1 || workers > 32 || !std::isfinite(limit) || limit <= 0 || limit > 60)
    throw py::value_error("Invalid TEASER input (3..4096 finite Nx3 points, positive bound, 1..32 workers)");
  for (Eigen::Index i = 0; i < source.rows(); ++i)
    for (Eigen::Index j = 0; j < i; ++j)
      if ((source.row(i) - source.row(j)).squaredNorm() == 0.)
        throw py::value_error("Coincident source locations create zero-length scale measurements");
  teaser::RegistrationSolution solution;
  solution.valid = false;
  solution.scale = std::numeric_limits<double>::quiet_NaN();
  solution.rotation.setConstant(std::numeric_limits<double>::quiet_NaN());
  solution.translation.setConstant(std::numeric_limits<double>::quiet_NaN());
  std::vector<int> clique, translation_inliers;
  double seconds;
  {
    py::gil_scoped_release release;
    ThreadLimit threads(workers);
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = bound;
    params.cbar2 = 1.;
    params.estimate_scaling = true;
    params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_gnc_factor = 1.4;
    params.rotation_max_iterations = 100;
    params.rotation_cost_threshold = 1e-6;
    params.rotation_tim_graph = teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN;
    params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;
    params.max_clique_num_threads = workers;
    params.max_clique_time_limit = limit;
    // The upstream solve mutates its rotation noise bound: never reuse a solver.
    teaser::RobustRegistrationSolver solver(params);
    const auto start = Clock::now();
    solver.solve(source.transpose(), target.transpose());
    seconds = std::chrono::duration<double>(Clock::now() - start).count();
    clique = solver.getInlierMaxClique();
    // A two-point clique cannot determine a full 3D rotation. The upstream
    // early-return path (clique <= 1) also leaves pose fields uninitialized.
    if (clique.size() >= 2) solution = solver.getSolution();
    if (solution.valid) translation_inliers = solver.getInputOrderedTranslationInliers();
  }
  py::dict out;
  out["valid"] = solution.valid;
  out["scale"] = solution.scale;
  out["rotation"] = Eigen::Matrix3d(solution.rotation);
  out["translation"] = Eigen::Vector3d(solution.translation);
  out["clique_indices"] = clique;
  out["translation_inlier_indices"] = translation_inliers;
  out["native_seconds"] = seconds;
  out["workers"] = workers;
  out["clique_time_limit_seconds"] = limit;
  out["clique_optimality_certified"] = false; // Upstream API exposes no timeout/proof flag.
  out["revision"] = "52a9c52ee7d4c838c5e8a75458c33178be5bfb70";
  return out;
}

PYBIND11_MODULE(_teaser, m) {
  m.doc() = "Optional upstream TEASER++ registration; no camera model";
  m.def("solve", &solve, py::arg("source"), py::arg("target"), py::arg("noise_bound"),
        py::arg("workers") = 4, py::arg("clique_time_limit") = 2.);
}
