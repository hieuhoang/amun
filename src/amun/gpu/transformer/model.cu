#include <iostream>
#include <sstream>
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
	Load(model, 1);
	Load(model, 2);
	Load(model, 3);
	Load(model, 4);
	Load(model, 5);
	Load(model, 6);

}

void WeightsTransformer::Load(const NpzConverter& model, unsigned num)
{
	cerr << "num=" << num << endl;
	stringstream ss;
	ss << num;
	std::string numStr = ss.str();

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_W1 = model.get("encoder_l" + numStr + "_ffn_W1", true);
	cerr << encoder_l1_ffn_W1.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_W2 = model.get("encoder_l" + numStr + "_ffn_W2", true);
	cerr << encoder_l1_ffn_W2.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_b1 = model.get("encoder_l" + numStr + "_ffn_b1", true);
	cerr << encoder_l1_ffn_b1.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_b2 = model.get("encoder_l" + numStr + "_ffn_b2", true);
	cerr << encoder_l1_ffn_b2.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_ffn_ln_bias = model.get("encoder_l" + numStr + "_ffn_ffn_ln_bias", true);
	cerr << encoder_l1_ffn_ffn_ln_bias.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_ffn_ffn_ln_scale = model.get("encoder_l" + numStr + "_ffn_ffn_ln_scale", true);
	cerr << encoder_l1_ffn_ffn_ln_scale.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wk = model.get("encoder_l" + numStr + "_self_Wk", true);
	cerr << encoder_l1_self_Wk.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wo = model.get("encoder_l" + numStr + "_self_Wo", true);
	cerr << encoder_l1_self_Wo.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wo_ln_bias = model.get("encoder_l" + numStr + "_self_Wo_ln_bias", true);
	cerr << encoder_l1_self_Wo_ln_bias.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wo_ln_scale = model.get("encoder_l" + numStr + "_self_Wo_ln_scale", true);
	cerr << encoder_l1_self_Wo_ln_scale.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wq = model.get("encoder_l" + numStr + "_self_Wq", true);
	cerr << encoder_l1_self_Wq.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_Wv = model.get("encoder_l" + numStr + "_self_Wv", true);
	cerr << encoder_l1_self_Wv.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_bk = model.get("encoder_l" + numStr + "_self_bk", true);
	cerr << encoder_l1_self_bk.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_bo = model.get("encoder_l" + numStr + "_self_bo", true);
	cerr << encoder_l1_self_bo.get()->Debug(1) << endl;

	std::shared_ptr<mblas::Tensor> encoder_l1_self_bq = model.get("encoder_l" + numStr + "_self_bq", true);
	cerr << encoder_l1_self_bq.get()->Debug(1) << endl;




}

}
}
