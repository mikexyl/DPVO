#include <teaser/registration.h>
#include <omp.h>
#include <random>
#include <iostream>
#include <stdexcept>

int main() {
  omp_set_num_threads(4);
  std::mt19937 rng(42);
  std::normal_distribution<double> normal;
  Eigen::Matrix<double,3,Eigen::Dynamic> a(3,100), b(3,100);
  for (int i=0;i<a.size();++i) a.data()[i]=normal(rng);
  Eigen::Matrix3d R=Eigen::Vector3d(-1,1,-1).asDiagonal();
  Eigen::Vector3d t(.5,-.2,.3);
  b=1.8*R*a;
  b.colwise()+=t;
  for (int i=20;i<100;++i)
    for (int d=0;d<3;++d) b(d,i)=3*normal(rng);
  for (int iteration=0;iteration<20;++iteration) {
    teaser::RobustRegistrationSolver::Params p;
    p.noise_bound=.03;
    p.max_clique_num_threads=4;
    p.max_clique_time_limit=2;
    teaser::RobustRegistrationSolver solver(p);
    solver.solve(a,b);
    auto s=solver.getSolution();
    if (!s.valid || std::abs(s.scale-1.8)>.03 || (s.rotation-R).norm()>.03 ||
        (s.translation-t).norm()>.03) throw std::runtime_error("known Sim3 recovery failed");
  }
  std::cout << "Native TEASER++ audit passed: repeated full-sphere 180-degree Sim3 with 80% outliers.\n";
}
