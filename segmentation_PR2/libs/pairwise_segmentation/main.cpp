#include <iostream>
#include "pairwise_segmentation.h"

int main()
{
  PairwiseSegmentation pw_seg;

  std::random_device generator;
  //  std::default_random_engine generator;
  std::uniform_int_distribution<int> size_distrib(0,5);
  std::uniform_int_distribution<int> moved_parts_distrib_1(1,5);
  std::uniform_int_distribution<int> moved_parts_distrib_2(6,10);
  for (int j = 0; j < 10; ++j)
  {
    std::cout << "-----------------NEW FRAME----------------\n";
    std::vector<uint32_t> moving_parts;
    int nb_of_moving_parts = size_distrib (generator);
    std::cout << "Number of parts moving: " << nb_of_moving_parts << "\n";
    for (int i = 0; i < nb_of_moving_parts; ++i)
    {
      uint32_t rd;
      if (j%2 == 0)
      { rd = static_cast<uint32_t> (moved_parts_distrib_1 (generator)); }
      else
      { rd = static_cast<uint32_t> (moved_parts_distrib_2 (generator)); }
      if (std::find (moving_parts.begin (), moving_parts.end (), rd)
          == moving_parts.end ())
      {
        moving_parts.push_back (rd);
        std::cout << rd << " ";
      }
      else
      { --i; }
    }
    std::cout << "\n";
    pw_seg.update (moving_parts, j);
    PairwiseSegmentation::Segmentation curr_seg =
        pw_seg.getCurrentSegmentation ();
    std::cout << "------------CURRENT CLUSTERING------------\n";
    for (const auto& pair: curr_seg)
    {
      std::cout << "[";
      for (const uint32_t label: pair.second)
      { std::cout << label << ", "; }
      std::cout << "\b\b] ";
    }
    std::cout << "\n";
  }
  return 0;
}
