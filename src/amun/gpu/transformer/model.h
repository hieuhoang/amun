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
			encoder_l1_ffn_W1,
			encoder_l1_ffn_W2,
			encoder_l1_ffn_b1,
			encoder_l1_ffn_b2,
			encoder_l1_ffn_ffn_ln_bias,
			encoder_l1_ffn_ffn_ln_scale,
			encoder_l1_self_Wk,
			encoder_l1_self_Wo,
			encoder_l1_self_Wo_ln_bias,
			encoder_l1_self_Wo_ln_scale,
			encoder_l1_self_Wq,
			encoder_l1_self_Wv,
			encoder_l1_self_bk,
			encoder_l1_self_bo,
			encoder_l1_self_bq;

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

