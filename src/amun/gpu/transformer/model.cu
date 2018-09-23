#include "model.h"

namespace amunmt {
namespace GPU {
	WeightsTransformer::WeightsTransformer(const std::string& npzFile, const YAML::Node& config,  unsigned device)
	: WeightsTransformer(NpzConverter(npzFile), config, device)
	{
	}


	WeightsTransformer::WeightsTransformer(const NpzConverter& model, const YAML::Node& config, unsigned device)
	{
		std::shared_ptr<mblas::Tensor> t = model.get("encoder_l1_ffn_W1", true);
		std::shared_ptr<mblas::Tensor> t2 = model.get("encoder_l1_ffn_W1sss", true);

	}

}
}
