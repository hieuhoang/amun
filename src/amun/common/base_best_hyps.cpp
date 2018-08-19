#include "base_best_hyps.h"
#include "god.h"

using namespace std;

namespace amunmt {

BaseBestHyps::BaseBestHyps(const God &god, unsigned maxBeamSize)
: god_(god),
  forbidUNK_(!god.Get<bool>("allow-unk")),
  isInputFiltered_(god.Get<std::vector<std::string>>("softmax-filter").size()),
  returnAttentionWeights_(god.Get<bool>("return-alignment") || god.Get<bool>("return-soft-alignment") || god.Get<bool>("return-nematus-alignment")),
  weights_(god.GetScorerWeights()),
  maxBeamSize_(maxBeamSize)
{}

}

