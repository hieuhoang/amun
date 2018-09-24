#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "../dl4mt/model.h"
#include "../npz_converter.h"

namespace amunmt {
namespace GPU {

class WeightsTransformer : public BaseWeights
{
	struct EncoderBunch
	{
		std::shared_ptr<mblas::Tensor>
			ffn_W1,
			ffn_W2,
			ffn_b1,
			ffn_b2,
			ffn_ffn_ln_bias,
			ffn_ffn_ln_scale,
			self_Wk,
			self_Wo,
			self_Wo_ln_bias,
			self_Wo_ln_scale,
			self_Wq,
			self_Wv,
			self_bk,
			self_bo,
			self_bq;

	};

public:
	WeightsTransformer(const std::string& npzFile, const YAML::Node& config,  unsigned device);
	WeightsTransformer(const NpzConverter& model, const YAML::Node& config, unsigned device);

protected:
	std::vector<EncoderBunch> encoderBunches_;

	void Load(const NpzConverter& model, unsigned num, EncoderBunch &bunch);
};

}
}

