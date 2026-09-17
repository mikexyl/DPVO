// Reuse the reference CBS factor construction and parameters, but instantiate
// exactly one robot's BPSAM. Python/ROS transports frozen beliefs between boards.
#define main cbs_reference_main
#include "cbs_dpvo_sim3_offline.cpp"
#undef main

ABSL_FLAG(std::string, agent_robot, "", "Only robot optimized by this process");

static void reply(const YAML::Node& value) {
  YAML::Emitter out;
  out.SetDoublePrecision(17);
  out << YAML::Flow << value;
  std::cout << "CBS_RPC " << out.c_str() << std::endl;
}

static YAML::Node encodeBeliefs(const std::map<Key, std::map<cbs::AgentId, gbp::Gaussian>>& beliefs) {
  YAML::Node rows(YAML::NodeType::Sequence);
  for (const auto& [key, origins] : beliefs) {
    for (const auto& [origin, belief] : origins) {
      require(belief.mu().allFinite() && belief.Sigma().allFinite(), "Non-finite outgoing belief");
      YAML::Node row;
      row["key"] = std::to_string(key);
      row["origin"] = static_cast<int>(origin);
      row["degree"] = belief.degree();
      for (int i = 0; i < belief.mu().size(); ++i) row["mu"].push_back(belief.mu()(i));
      const auto sigma = belief.Sigma();
      for (int i = 0; i < sigma.rows(); ++i)
        for (int j = 0; j < sigma.cols(); ++j) row["sigma"].push_back(sigma(i,j));
      rows.push_back(row);
    }
  }
  return rows;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  try {
    const Dataset dataset = readDataset(absl::GetFlag(FLAGS_input_graph));
    const auto robot = absl::GetFlag(FLAGS_agent_robot);
    require(dataset.robot_to_agent.count(robot) == 1, "Agent not in graph");
    const size_t agent = dataset.robot_to_agent.at(robot);
    OfflineCbs runtime(dataset, robot);
    require(runtime.sams_.size() == 1, "Distributed process must own exactly one BPSAM");
    auto& sam = *runtime.sams_.at(agent);
    YAML::Node ready;
    ready["ok"] = true; ready["robot"] = robot; ready["optimizer_instances"] = runtime.sams_.size();
    ready["neighbors"] = YAML::Node(YAML::NodeType::Sequence);
    for (size_t peer : runtime.neighbor_agents_.at(agent)) ready["neighbors"].push_back(dataset.robot_ids.at(peer));
    reply(ready);
    std::string line;
    while (std::getline(std::cin, line)) {
      const auto request = YAML::Load(line);
      const auto op = request["op"].as<std::string>();
      YAML::Node response; response["ok"] = true;
      if (op == "snapshot") {
        const bool anchor = request["anchor"].as<bool>();
        KeySet keys = runtime.separator_keys_;
        if (anchor) {
          keys.clear();
          for (size_t i = 0; i < dataset.robot_ids.size(); ++i) keys.insert(cbs::toRobotKey(toCbsAgent(i)));
        }
        sam.setMarginalizationGraph(anchor ? OfflineCbs::Sam::MarginalizationType::GLOBAL
                                           : OfflineCbs::Sam::MarginalizationType::LOCAL);
        response["beliefs"] = encodeBeliefs(sam.getBeliefs(keys, true));
      } else if (op == "update") {
        size_t count = 0;
        for (const auto& packet : request["packets"]) {
          const auto sender = packet["sender"].as<std::string>();
          require(dataset.robot_to_agent.count(sender) &&
                  runtime.neighbor_agents_.at(agent).count(dataset.robot_to_agent.at(sender)), "Unexpected belief sender");
          std::map<Key, std::map<cbs::AgentId, gbp::Gaussian>> beliefs;
          for (const auto& row : packet["beliefs"]) {
            const Key key = std::stoull(row["key"].as<std::string>());
            if (!cbs::isRobotKey(gtsam::LabeledSymbol(key)) && !runtime.receivable_pose_keys_.at(agent).count(key)) continue;
            require(row["mu"].size() == 7 && row["sigma"].size() == 49, "Invalid Sim3 belief dimensions");
            gtsam::Vector mu(7); gtsam::Matrix sigma(7,7);
            for (int i = 0; i < 7; ++i) {
              mu(i) = row["mu"][i].as<double>();
              for (int j = 0; j < 7; ++j) sigma(i,j) = row["sigma"][i*7+j].as<double>();
            }
            require(mu.allFinite() && sigma.allFinite(), "Non-finite received belief");
            beliefs[key].emplace(static_cast<cbs::AgentId>(row["origin"].as<int>()),
                                gbp::Gaussian(key, mu, sigma, row["degree"].as<size_t>()));
            ++count;
          }
          sam.addBeliefs(std::move(beliefs));
        }
        sam.update({}, {}, OfflineCbs::Sam::UpdateParams{});
        response["received_beliefs"] = count;
      } else if (op == "result" || op == "preview") {
        const auto values = sam.calculateEstimate();
        const auto anchor = cbs::toRobotKey(toCbsAgent(0));
        const auto own = cbs::toRobotKey(toCbsAgent(agent));
        require(values.exists(own), "Own map estimate is missing");
        if (op == "result")
          require(values.exists(anchor), "Anchor belief has not reached this agent");
        const auto first = firstPoseKey(dataset, agent);
        // Provisional estimates use the common initialization gauge. Reading them
        // must not require anchor beliefs to have propagated through the graph.
        const auto reference = op == "preview" ? Similarity3::Identity()
                                                : values.at<Similarity3>(anchor).inverse();
        const auto map = reference * values.at<Similarity3>(own)
                       * values.at<Similarity3>(first) * dataset.poses.at(first).estimate.inverse();
        const auto t = physicalTranslation(map);
        const auto q = map.rotation().toQuaternion();
        for (int i = 0; i < 3; ++i) response["translation"].push_back(t(i));
        for (double v : {q.x(), q.y(), q.z(), q.w()}) response["quaternion"].push_back(v);
        response["scale"] = map.scale();
      } else if (op == "quit") { break; }
      else throw std::runtime_error("Unknown agent operation");
      reply(response);
    }
    return 0;
  } catch (const std::exception& error) {
    YAML::Node response; response["ok"] = false; response["error"] = error.what();
    reply(response); return 1;
  }
}
