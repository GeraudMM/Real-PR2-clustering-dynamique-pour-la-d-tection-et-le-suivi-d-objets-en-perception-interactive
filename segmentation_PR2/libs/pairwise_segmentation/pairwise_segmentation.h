#ifndef PAIRWISE_SEGMENTATION_H
#define PAIRWISE_SEGMENTATION_H

#include <iostream>
#include <unordered_map>
#include <map>
#include <vector>
#include <algorithm>
#include <set>

class PairwiseSegmentation
{
  public:
    typedef std::unordered_map<uint32_t, std::vector<uint32_t>> Segmentation;
    typedef std::unordered_map <uint32_t, int> Counter;
    typedef std::unordered_map<uint32_t, Counter> GlobalCounter;
    PairwiseSegmentation ();
    /**
     * @brief PairwiseSegmentation
     * @param moving_parts_labels
     */
    PairwiseSegmentation (const std::vector<uint32_t>& moving_parts_labels);
    /**
     * @brief update
     * @param moving_parts_labels vector containing the label of the parts that
     * were observed moving this frame
     */
    void
    update (const std::vector<uint32_t>& moving_parts_labels,
            const int frame_count);
    /**
     * @brief resetPart
     * @param part_label
     */
    void
    resetPart (uint32_t part_label);
    /**
     * @brief resetParts
     * @param parts_labels
     */
    void
    resetParts (const std::vector<uint32_t> parts_labels);
    /**
     * @brief getCurrentSegmentation
     * @return the current segmentation, which is an std::unordered_map with
     * keys being the label of the potential object and values being the
     * labels of the parts belonging to this object
     */
    const Segmentation&
    getCurrentSegmentation () const
    { return current_segmentation_; }
  private:
    /**
     * @brief removeDuplicates
     */
    void
    removeDuplicates ();
    /**
     * @brief regroupObjects
     */
    void
    regroupObjects ();
    /**
     * @brief computeObjectCandidates
     */
    void
    computeObjectCandidates ();
    /**
     * @brief updateComovementCounts
     * @param moving_parts_labels
     */
    void
    updateComovementCounts (const std::vector<uint32_t>& moving_parts_labels,
                            const int frame_count);
    GlobalCounter comovement_counts_;
    /**
     * @brief current_segmentation_ current grouping of parts
     */
    Segmentation current_segmentation_;
};

#ifndef PAIRWISE_SEGMENTATION_HPP
#include "pairwise_segmentation.hpp"
#endif
#endif // PAIRWISE_SEGMENTATION_H
