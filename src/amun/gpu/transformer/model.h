#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "../dl4mt/model.h"

namespace amunmt {
namespace GPU {

class WeightsTransformer : public BaseWeights
{
public:
	WeightsTransformer(const std::string& npzFile, const YAML::Node& config,  unsigned device);

};

}
}

