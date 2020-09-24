#ifndef PCL_TRACKING_IMPL_LABEL_COHERENCE_HPP_
#define PCL_TRACKING_IMPL_LABEL_COHERENCE_HPP_

#include <pcl/common/common.h>
#include <pcl/console/print.h>
#include "../label_coherence.h"

template <typename PointT> double 
pcl::tracking::LabelCoherence<PointT>::computeCoherence (PointT &source, PointT &target)
{
  if(source.label == target.label)
  {
    return 1.0;
  }
  else
  {
//    std::cout << "Return 0.1\n";
    return 0.1;
  }
}


#define PCL_INSTANTIATE_LabelCoherence(T) template class PCL_EXPORTS pcl::tracking::LabelCoherence<T>;

#endif
