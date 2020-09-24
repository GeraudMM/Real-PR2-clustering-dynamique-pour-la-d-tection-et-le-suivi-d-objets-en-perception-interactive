#ifndef PCL_TRACKING_LABEL_COHERENCE_H_
#define PCL_TRACKING_LABEL_COHERENCE_H_

#include <pcl/tracking/coherence.h>
namespace pcl
{
  namespace tracking
  {
    /** \brief @b LabelCoherence computes coherence between two points from the labelization
      * of those two points
      * \author Elias Hanna
      * \ingroup tracking
      */
    template <typename PointT>
    class LabelCoherence: public PointCoherence<PointT>
    {
    public:

      /** \brief initialize the weight to 1.0. */
      LabelCoherence ()
      : PointCoherence<PointT> ()
      , weight_ (1.0)
        {}

      /** \brief set the weight of coherence
        * \param weight the weight of coherence
        */
      inline void setWeight (double weight) { weight_ = weight; }

      /** \brief get the weight of coherence */
      inline double getWeight () { return weight_; }

    protected:

      /** \brief return the label coherence between the two points.
        * \param source instance of source point.
        * \param target instance of target point.
        */
      double computeCoherence (PointT &source, PointT &target);

      /** \brief the weight of coherence */
      double weight_;
      
    };
  }
}

#ifdef PCL_NO_PRECOMPILE
#include "impl/label_coherence.hpp"
#endif

#endif
