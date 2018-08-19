#pragma once

#include <functional>
#include <vector>
#include <map>

#include "common/types.h"
#include "scorer.h"

namespace amunmt {

class Histories;

class BaseBestHyps
{
  public:
    BaseBestHyps(const God &god, unsigned maxBeamSize);

    BaseBestHyps(const BaseBestHyps&) = delete;

    virtual void CalcBeam(
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams) = 0;

    virtual void BeginSentenceState(unsigned batchSize) = 0;
    virtual const void* GetBeamSizes() const = 0;

    template<class T>
    const T &GetBeamSizes() const
    {
      const void *t = GetBeamSizes();
      const T &ret = *static_cast<const T*>(t);
      return ret;
    }

    virtual bool CalcBeam(
        const std::vector<ScorerPtr>& scorers,
        const Words &filterIndices,

        std::shared_ptr<Histories>& histories,
        Beam& prevHyps,
        States& states,
        States& nextStates,
        unsigned decoderStep) = 0;


  protected:
    const God &god_;
    const bool forbidUNK_;
    const bool isInputFiltered_;
    const bool returnAttentionWeights_;
    const std::map<std::string, float> weights_;
    const unsigned maxBeamSize_;

};

typedef std::shared_ptr<BaseBestHyps> BaseBestHypsPtr;

}
