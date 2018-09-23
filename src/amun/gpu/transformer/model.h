#pragma once

#include <string>
#include <yaml-cpp/yaml.h>

namespace amunmt {
namespace GPU {

class WeightsTransformer
{
public:
	WeightsTransformer(const std::string& npzFile, const YAML::Node& config,  unsigned device);

};

}
}

