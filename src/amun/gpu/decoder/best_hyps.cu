#include "best_hyps.h"
#include "common/beam.h"
#include "common/history.h"
#include "common/histories.h"

using namespace std;

namespace amunmt {
namespace GPU {

BestHyps::BestHyps(const God &god, unsigned maxBeamSize)
      : BaseBestHyps(god, maxBeamSize),
        keys_(god.Get<unsigned>("beam-size") * god.Get<unsigned>("mini-batch")),
        costs_(god.Get<unsigned>("beam-size") * god.Get<unsigned>("mini-batch")),
        maxBeamSize_(god.Get<unsigned>("beam-size"))
{
  if (!god_.UseFusedSoftmax()) {
    NthElement *obj = new NthElement(god.Get<unsigned>("beam-size"), god.Get<unsigned>("mini-batch"));
    nthElement_.reset(obj);
  }
}

void BestHyps::BeginSentenceState(unsigned batchSize)
{
  beamSizes_.clear();
  beamSizes_.resize(batchSize, 1);
}

const void* BestHyps::GetBeamSizes() const
{
  return &beamSizes_;
}

void BestHyps::DisAllowUNK(mblas::Tensor& Prob) {
  SetColumn(Prob, UNK_ID, std::numeric_limits<float>::lowest());
}

void BestHyps::FindBests(mblas::Tensor& Probs,
                         std::vector<float>& outCosts,
                         std::vector<unsigned>& outKeys,
                         const bool isFirst)
{
  nthElement_->getNBestList(beamSizes_, Probs, outCosts, outKeys, isFirst);
}

// fast fused softmax and nth_element
void BestHyps::FindBests(mblas::Tensor& Probs,
                         mblas::Vector<NthOutBatch> &nBest,
                         std::vector<float>& outCosts,
                         std::vector<unsigned>& outKeys,
                         const bool isFirst)
{
  getNBestList(Probs, nBest, outCosts, outKeys, isFirst);
}

std::vector<SoftAlignmentPtr> BestHyps::GetAlignments(const std::vector<ScorerPtr>& scorers,
                                            unsigned hypIndex)
{
  std::vector<SoftAlignmentPtr> alignments;
  for (auto& scorer : scorers) {
    if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorer.get())) {
      const mblas::Tensor &attention = encdec->GetAttention();
      unsigned attLength = attention.dim(1);

      SoftAlignment *softAlignment = new SoftAlignment(attLength);
      mblas::copy(
          attention.data() + hypIndex * attLength,
          attLength,
          softAlignment->data(),
          cudaMemcpyDeviceToHost
      );

      alignments.emplace_back(softAlignment);
    } else {
      amunmt_UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
    }
  }
  return alignments;
}

// standard nth_element
void  BestHyps::CalcBeam(
    const Beam& prevHyps,
    const std::vector<ScorerPtr>& scorers,
    const Words& filterIndices,
    std::vector<Beam>& beams)
{
  BEGIN_TIMER("CalcBeam");

  using namespace mblas;

  mblas::Tensor& Probs = static_cast<mblas::Tensor&>(scorers[0]->GetProbs());

  std::vector<float> vCosts;
  for (auto& h : prevHyps) {
    vCosts.push_back(h->GetCost());
  }
  const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

  if (isFirst) {
    for (auto& beamSize : beamSizes_) {
      beamSize = maxBeamSize_;
    }
  }

  mblas::copy(vCosts.data(),
              vCosts.size(),
              costs_.data(),
              cudaMemcpyHostToDevice);
  //mblas::copy(vCosts.begin(), vCosts.end(), costs_.begin());

  unsigned beamSizeSum = std::accumulate(beamSizes_.begin(), beamSizes_ .end(), 0);

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;


  if (god_.UseFusedSoftmax()) {
    const mblas::Tensor& b4 = *static_cast<const mblas::Tensor*>(scorers[0]->GetBias());
    mblas::Vector<NthOutBatch> &nBest = *static_cast<mblas::Vector<NthOutBatch>*>(scorers[0]->GetNBest());
    nBest.newSize(beamSizeSum);

    bool requireProb = maxBeamSize_ > 1 || god_.Get<bool>("n-best");
    //cerr << "doSoftmax=" << doSoftmax << endl;

    BEGIN_TIMER("GetProbs.LogSoftmaxAndNBest");
    mblas::LogSoftmaxAndNBest(nBest, Probs, b4, costs_, forbidUNK_, maxBeamSize_, beamSizes_, beamSizeSum, isFirst, requireProb);
    PAUSE_TIMER("GetProbs.LogSoftmaxAndNBest");
    //std::cerr << "2Probs=" << Probs.Debug(1) << std::endl;

    FindBests(Probs, nBest, bestCosts, bestKeys, isFirst);
  }
  else {
    BroadcastVecColumn(weights_.at(scorers[0]->GetName()) * _1 + _2, Probs, costs_);

    for (unsigned i = 1; i < scorers.size(); ++i) {
      mblas::Tensor &currProbs = static_cast<mblas::Tensor&>(scorers[i]->GetProbs());

      Element(_1 + weights_.at(scorers[i]->GetName()) * _2, Probs, currProbs);
    }

    if (forbidUNK_) {
      DisAllowUNK(Probs);
    }

    FindBests(Probs, bestCosts, bestKeys, isFirst);
  }

  std::vector<std::vector<float>> breakDowns;
  if (god_.ReturnNBestList()) {
      breakDowns.push_back(bestCosts);
      for (unsigned i = 1; i < scorers.size(); ++i) {
        std::vector<float> modelCosts(beamSizeSum);
        mblas::Tensor &currProbs = static_cast<mblas::Tensor&>(scorers[i]->GetProbs());

        nthElement_->getValueByKey(modelCosts, currProbs);
        breakDowns.push_back(modelCosts);
      }
  }

  std::map<unsigned, unsigned> batchMap;
  unsigned tmp = 0;
  for (unsigned batchID = 0; batchID < beamSizes_.size(); ++batchID) {
    for (unsigned t = 0; t < beamSizes_[batchID]; ++t) {
      batchMap[tmp++] = batchID;
    }
  }

  for (unsigned i = 0; i < beamSizeSum; i++) {
    unsigned wordIndex = bestKeys[i] % Probs.dim(1);
    if (isInputFiltered_) {
      wordIndex = filterIndices[wordIndex];
    }

    unsigned hypIndex  = bestKeys[i] / Probs.dim(1);
    float cost = bestCosts[i];

    HypothesisPtr hyp;
    if (returnAttentionWeights_) {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
                               GetAlignments(scorers, hypIndex)));
    } else {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
    }

    //cerr << "god_.ReturnNBestList()=" << god_.ReturnNBestList() << endl;
    if(god_.ReturnNBestList()) {
      hyp->GetCostBreakdown().resize(scorers.size());
      float sum = 0;
      for (unsigned j = 0; j < scorers.size(); ++j) {
        if (j == 0)
          hyp->GetCostBreakdown()[0] = breakDowns[0][i];
        else {
          float cost = 0;
          if (j < scorers.size()) {
              if (prevHyps[hypIndex]->GetCostBreakdown().size() < scorers.size())
                const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(scorers.size(), 0.0f);
              cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
          }
          sum += weights_.at(scorers[j]->GetName()) * cost;
          hyp->GetCostBreakdown()[j] = cost;
        }
      }
      hyp->GetCostBreakdown()[0] -= sum;
      hyp->GetCostBreakdown()[0] /= weights_.at(scorers[0]->GetName());
    }

    beams[batchMap[i]].push_back(hyp);
  }

  PAUSE_TIMER("CalcBeam");
}

//////////////////////////////////////////////////////////////////////////
void BestHyps::getNBestList(
                  mblas::Tensor& Probs,
                  mblas::Vector<NthOutBatch> &nBest,
                  std::vector<float>& outCosts,
                  std::vector<unsigned>& outKeys,
                  const bool isFirst) const
{
  GetPairs(nBest, outKeys, outCosts);
  assert(outCosts.size() == outKeys.size());

  /*
  cerr << "outCosts/outKeys=";
  for (unsigned i = 0; i < outKeys.size(); ++i) {
    cerr << "(" << outCosts[i] << "," << outKeys[i] << ") ";
  }
  cerr << endl;
  */
  //cerr << endl;
}

void BestHyps::GetPairs(mblas::Vector<NthOutBatch> &nBest,
              std::vector<unsigned>& outKeys,
              std::vector<float>& outValues) const
{
  //cerr << "top=" << top2.size() << " nBest=" << nBest.size() << endl;
  outKeys.resize(nBest.size());
  outValues.resize(nBest.size());

  std::vector<NthOutBatch> hostVec(nBest.size());
  mblas::copy(nBest.data(), nBest.size(), hostVec.data(), cudaMemcpyDeviceToHost);

  for (unsigned i = 0; i < nBest.size(); ++i) {
    outKeys[i] = hostVec[i].ind;
    outValues[i] = hostVec[i].score;
  }
}

bool BestHyps::CalcBeam(
    const std::vector<ScorerPtr>& scorers,
    const Words &filterIndices,

    std::shared_ptr<Histories>& histories,
    Beam& prevHyps,
    States& states,
    States& nextStates,
    unsigned decoderStep)
{
    unsigned batchSize = beamSizes_.size();
    Beams beams(batchSize);
    CalcBeam(prevHyps, scorers, filterIndices, beams);
    histories->Add(beams);

    //cerr << "batchSize=" << batchSize << endl;
    histories->SetActive(false);
    Beam survivors;
    for (unsigned batchId = 0; batchId < batchSize; ++batchId) {
      const History &hist = *histories->at(batchId);
      unsigned maxLength = hist.GetMaxLength();

      //cerr << "beamSizes[batchId]=" << batchId << " " << beamSizes[batchId] << " " << maxLength << endl;
      for (auto& h : beams[batchId]) {
        if (decoderStep < maxLength && h->GetWord() != EOS_ID) {
          survivors.push_back(h);

          histories->SetActive(batchId, true);
        } else {
          --beamSizes_[batchId];
        }
      }
    }

    if (survivors.size() == 0) {
      return false;
    }

    for (unsigned i = 0; i < scorers.size(); i++) {
      scorers[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    //cerr << "survivors=" << survivors.size() << endl;
    prevHyps.swap(survivors);
    return true;
}

} // namespace
}
