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
: encoderBunches_(6)
{
	Load(model, 1, encoderBunches_[0]);
	Load(model, 2, encoderBunches_[1]);
	Load(model, 3, encoderBunches_[2]);
	Load(model, 4, encoderBunches_[3]);
	Load(model, 5, encoderBunches_[4]);
	Load(model, 6, encoderBunches_[5]);

}

void WeightsTransformer::Load(const NpzConverter& model, unsigned num, EncoderBunch &bunch)
{
	cerr << "num=" << num << endl;
	stringstream ss;
	ss << num;
	std::string numStr = ss.str();

	bunch.ffn_W1 = model.get("encoder_l" + numStr + "_ffn_W1", true);
	cerr << bunch.ffn_W1.get()->Debug(1) << endl;

	bunch.ffn_W2 = model.get("encoder_l" + numStr + "_ffn_W2", true);
	cerr << bunch.ffn_W2.get()->Debug(1) << endl;

	bunch.ffn_b1 = model.get("encoder_l" + numStr + "_ffn_b1", true);
	cerr << bunch.ffn_b1.get()->Debug(1) << endl;

	bunch.ffn_b2 = model.get("encoder_l" + numStr + "_ffn_b2", true);
	cerr << bunch.ffn_b2.get()->Debug(1) << endl;

	bunch.ffn_ffn_ln_bias = model.get("encoder_l" + numStr + "_ffn_ffn_ln_bias", true);
	cerr << bunch.ffn_ffn_ln_bias.get()->Debug(1) << endl;

	bunch.ffn_ffn_ln_scale = model.get("encoder_l" + numStr + "_ffn_ffn_ln_scale", true);
	cerr << bunch.ffn_ffn_ln_scale.get()->Debug(1) << endl;

	bunch.self_Wk = model.get("encoder_l" + numStr + "_self_Wk", true);
	cerr << bunch.self_Wk.get()->Debug(1) << endl;

	bunch.self_Wo = model.get("encoder_l" + numStr + "_self_Wo", true);
	cerr << bunch.self_Wo.get()->Debug(1) << endl;

	bunch.self_Wo_ln_bias = model.get("encoder_l" + numStr + "_self_Wo_ln_bias", true);
	cerr << bunch.self_Wo_ln_bias.get()->Debug(1) << endl;

	bunch.self_Wo_ln_scale = model.get("encoder_l" + numStr + "_self_Wo_ln_scale", true);
	cerr << bunch.self_Wo_ln_scale.get()->Debug(1) << endl;

	bunch.self_Wq = model.get("encoder_l" + numStr + "_self_Wq", true);
	cerr << bunch.self_Wq.get()->Debug(1) << endl;

	bunch.self_Wv = model.get("encoder_l" + numStr + "_self_Wv", true);
	cerr << bunch.self_Wv.get()->Debug(1) << endl;

	bunch.self_bk = model.get("encoder_l" + numStr + "_self_bk", true);
	cerr << bunch.self_bk.get()->Debug(1) << endl;

	bunch.self_bo = model.get("encoder_l" + numStr + "_self_bo", true);
	cerr << bunch.self_bo.get()->Debug(1) << endl;

	bunch.self_bq = model.get("encoder_l" + numStr + "_self_bq", true);
	cerr << bunch.self_bq.get()->Debug(1) << endl;




}

}
}
