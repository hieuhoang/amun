#include <iostream>
#include "model.h"
#include "gpu/mblas/tensor.h"

using namespace std;

namespace amunmt {
namespace GPU {

WeightsTransformer::WeightsTransformer(const std::string& npzFile, const YAML::Node& config,  unsigned device)
: WeightsTransformer(NpzConverter(npzFile), config, device)
{
}


WeightsTransformer::WeightsTransformer(const NpzConverter& model, const YAML::Node& config, unsigned device)
{
	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_W1 = model.get("encoder_l1_ffn_W1", true);
	cerr << encoder_l1_ffn_W1.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_W2 = model.get("encoder_l1_ffn_W2", true);
	cerr << encoder_l1_ffn_W2.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_b1 = model.get("encoder_l1_ffn_b1", true);
	cerr << encoder_l1_ffn_b1.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_b2 = model.get("encoder_l1_ffn_b2", true);
	cerr << encoder_l1_ffn_b2.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_ffn_ln_bias = model.get("encoder_l1_ffn_ffn_ln_bias", true);
	cerr << encoder_l1_ffn_ffn_ln_bias.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_ffn_ln_scale = model.get("encoder_l1_ffn_ffn_ln_scale", true);
	cerr << encoder_l1_ffn_ffn_ln_scale.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wk = model.get("encoder_l1_self_Wk", true);
	cerr << encoder_l1_self_Wk.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wo = model.get("encoder_l1_self_Wo", true);
	cerr << encoder_l1_self_Wo.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wo_ln_bias = model.get("encoder_l1_self_Wo_ln_bias", true);
	cerr << encoder_l1_self_Wo_ln_bias.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wo_ln_scale = model.get("encoder_l1_self_Wo_ln_scale", true);
	cerr << encoder_l1_self_Wo_ln_scale.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wq = model.get("encoder_l1_self_Wq", true);
	cerr << encoder_l1_self_Wq.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wv = model.get("encoder_l1_self_Wv", true);
	cerr << encoder_l1_self_Wv.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_bk = model.get("encoder_l1_self_bk", true);
	cerr << encoder_l1_self_bk.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_bo = model.get("encoder_l1_self_bo", true);
	cerr << encoder_l1_self_bo.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_bq = model.get("encoder_l1_self_bq", true);
	cerr << encoder_l1_self_bq.get()->Debug(1) << endl;




}

}
}
