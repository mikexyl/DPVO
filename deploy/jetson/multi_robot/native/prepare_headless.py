"""Remove only the unused C++ Rerun frontend from an isolated CBS source copy.

The ROS/Viser coordinator consumes solver CSVs; its solver never enables Rerun.
CBS estimation code and tests remain unchanged.
"""
from pathlib import Path
import sys
root = Path(sys.argv[1])
p = root / 'examples/cbs_dpvo_sim3_offline.cpp'
s = p.read_text()
a = s.index('class RerunVisualization {')
b = s.index('\nvoid validateFlags()', a)
s = s[:a] + '''class RerunVisualization {
 public:
  explicit RerunVisualization(const std::filesystem::path&) {
    if (absl::GetFlag(FLAGS_write_rerun_rrd) || absl::GetFlag(FLAGS_rerun_stream))
      throw std::runtime_error("This headless build uses the ROS/Viser viewer");
  }
  bool enabled() const { return false; }
  bool factorGraphsEnabled() const { return false; }
  template<class... T> void logFactorGraph(T&&...) {}
  template<class... T> void beginConvergence(T&&...) {}
  template<class... T> void logCbsIteration(T&&...) {}
  template<class... T> void logSolution(T&&...) {}
  template<class... T> void logAnchors(T&&...) {}
  void finishConvergence() {}
  void flush() {}
  std::filesystem::path rrdPath() const { return {}; }
};
''' + s[b:]
s = s.replace('#include <aria_viz/visualizer_rerun.h>', '')
# Expose the reference initialization to the wire adapter and select one agent.
a = s.index('class OfflineCbs {')
b = s.index('\nValues assembleCbsTrajectory(', a)
block = s[a:b].replace('explicit OfflineCbs(const Dataset& dataset)',
    'explicit OfflineCbs(const Dataset& dataset, const std::string& selected_robot = "")')
block = block.replace(': dataset_(dataset),', ': dataset_(dataset), selected_robot_(selected_robot),')
block = block.replace(' private:', ' public:', 1)
block = block.replace('  const Dataset& dataset_;', '  const Dataset& dataset_;\n  std::string selected_robot_;')
block = block.replace('      typename Sam::Params params;',
    '      if (!selected_robot_.empty() && dataset_.robot_ids[agent] != selected_robot_) continue;\n      typename Sam::Params params;')
s = s[:a] + block + s[b:]
p.write_text(s)
p = root / 'CMakeLists.txt'
s = p.read_text().replace('add_subdirectory(src/utils)', '')
p.write_text(s)
(root / 'examples/CMakeLists.txt').write_text('''find_package(absl CONFIG REQUIRED)
add_executable(cbs_dpvo_sim3_offline cbs_dpvo_sim3_offline.cpp)
target_link_libraries(cbs_dpvo_sim3_offline PRIVATE cbs yaml-cpp absl::flags absl::flags_parse)
add_executable(cbs_agent cbs_agent.cpp)
target_link_libraries(cbs_agent PRIVATE cbs yaml-cpp absl::flags absl::flags_parse)
''')
