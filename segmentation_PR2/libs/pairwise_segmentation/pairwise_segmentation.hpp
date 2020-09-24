#ifndef PAIRWISE_SEGMENTATION_HPP
#define PAIRWISE_SEGMENTATION_HPP

#include "pairwise_segmentation.h"

///////////////////////////////////////////////////////////////////////////////
template <class Container>
auto findMaxValuePair(Container const &x)
-> typename Container::value_type
{
  using value_t = typename Container::value_type;
  const auto compare = [](value_t const &p1, value_t const &p2)
  {
    return p1.second < p2.second;
  };
  return *std::max_element(x.begin(), x.end(), compare);
}

///////////////////////////////////////////////////////////////////////////////
PairwiseSegmentation::PairwiseSegmentation ():
  comovement_counts_ (),
  current_segmentation_ ()
{

}

///////////////////////////////////////////////////////////////////////////////
PairwiseSegmentation::PairwiseSegmentation (const std::vector<uint32_t>&
                                            moving_parts_labels):
  comovement_counts_ (),
  current_segmentation_ ()
{
  update (moving_parts_labels, 0);
}

///////////////////////////////////////////////////////////////////////////////
void
PairwiseSegmentation::updateComovementCounts (const std::vector<uint32_t>&
                                              moving_parts_labels,
                                              const int frame_count)
{
  for (const uint32_t moving_part_label: moving_parts_labels)
  {
    GlobalCounter::const_iterator global_counter_it =
        comovement_counts_.find (moving_part_label);

    // Initialization
    if (global_counter_it == comovement_counts_.end ())
    {
      comovement_counts_.insert (std::pair <uint32_t, Counter>
                                 (moving_part_label, Counter ()));
    }
    Counter& moving_part_counter = comovement_counts_.at (moving_part_label);
    // Handle no comovement
    for (auto& other_part: moving_part_counter)
    {
      // Check if comovement was observed before but not at this frame
      if (std::find
          (moving_parts_labels.begin (), moving_parts_labels.end (),
           other_part.first)
          == moving_parts_labels.end ())
      {
        --other_part.second;
        --comovement_counts_.at (other_part.first).at (moving_part_label);
      }
    }
    // Handle comovement
    for (const uint32_t other_part_label: moving_parts_labels)
    {
      if (other_part_label != moving_part_label)
      {
        Counter::const_iterator counter_it =
            moving_part_counter.find (other_part_label);

        bool flag = true;
        if (counter_it == moving_part_counter.end ())
        {
          if(comovement_counts_.find (other_part_label)
             != comovement_counts_.end ())
          {
            if (comovement_counts_.at (other_part_label).find (moving_part_label)
                != comovement_counts_.at (other_part_label). end ())
            {
              if (comovement_counts_.at (other_part_label).at(moving_part_label)
                  != 1)
              {
                moving_part_counter.insert
                    (std::pair <uint32_t, int> (other_part_label, 1 - frame_count));
                flag = false;
              }
            }
          }
          if (flag)
          {
            moving_part_counter.insert
                (std::pair <uint32_t, int> (other_part_label, 1));
          }
        }
        else
        {
          ++(moving_part_counter.at (other_part_label));
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
void
PairwiseSegmentation::computeObjectCandidates ()
{
  int nb_obj = 0;
  std::vector<uint32_t> copy_vec;
  for (const auto& pair: comovement_counts_)
  {
    copy_vec.push_back (pair.first);
  }
  while (copy_vec.begin () != copy_vec.end ())
  {
    uint32_t moving_part_label = *(copy_vec.begin());
    ++nb_obj;
    current_segmentation_.insert (std::pair<uint32_t, std::vector<uint32_t>>
                                  (nb_obj, std::vector<uint32_t> ()));
    current_segmentation_.at (nb_obj).push_back (moving_part_label);
    copy_vec.erase(std::remove
                   (copy_vec.begin(), copy_vec.end(), moving_part_label),
                   copy_vec.end());
    if (comovement_counts_.at (moving_part_label).size() > 0)
    {
      for (const auto& pair: comovement_counts_.at (moving_part_label))
      {
        if (pair.second > 0)
        {
          current_segmentation_.at (nb_obj).push_back (pair.first);

          copy_vec.erase(std::remove
                         (copy_vec.begin(), copy_vec.end(), pair.first),
                         copy_vec.end());
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
void
PairwiseSegmentation::regroupObjects ()
{
  Segmentation::iterator seg_it = current_segmentation_.begin();
  int it = 0; int it2 = 0;
  while (seg_it != current_segmentation_.end ())
  {
    std::pair<uint32_t, std::vector<uint32_t>> pair = (*seg_it);
    Segmentation::iterator seg_other_it = current_segmentation_.begin ();
    while (seg_other_it != current_segmentation_.end ())
    {
      std::pair<uint32_t, std::vector<uint32_t>> other = (*seg_other_it);
      bool breaked = false;
      if (pair.first != other.first)
      {
        for (const auto& obj_1: pair.second)
        {
          if (!breaked)
          {
            for (const auto& obj_2: other.second)
            {
              // Merge the objects
              if (obj_1 == obj_2)
              {
                (*seg_it).second.insert ((*seg_it).second.end (),
                                         other.second.begin (), other.second.end ());
                breaked = true; break;
              }
            }
          }
          else { break; }
        }
      }
      if (breaked) { current_segmentation_.erase (seg_other_it++); }
      else { ++seg_other_it; }
      ++it2;
    }
    ++seg_it;
    it++;
  }
}

///////////////////////////////////////////////////////////////////////////////
void
PairwiseSegmentation::removeDuplicates ()
{
  for (auto& pair: current_segmentation_)
  {
    sort(pair.second.begin(), pair.second.end());
    pair.second.erase (std::unique ( pair.second.begin(), pair.second.end ()),
                       pair.second.end ());
  }
}

///////////////////////////////////////////////////////////////////////////////
void
PairwiseSegmentation::resetPart (uint32_t part_label)
{
  comovement_counts_.erase (part_label);
  for (auto& pair: comovement_counts_)
  {
    pair.second.erase (part_label);
  }
}

///////////////////////////////////////////////////////////////////////////////
void
PairwiseSegmentation::resetParts (const std::vector<uint32_t> parts_labels)
{
  for (const uint32_t label: parts_labels)
  { resetPart (label); }
}

///////////////////////////////////////////////////////////////////////////////
void
PairwiseSegmentation::update (const std::vector<uint32_t>& moving_parts_labels,
                              const int frame_count)
{
  current_segmentation_.clear ();

  updateComovementCounts (moving_parts_labels, frame_count);

  computeObjectCandidates ();

  regroupObjects ();

  removeDuplicates ();
}

#endif // PAIRWISE_SEGMENTATION_HPP
