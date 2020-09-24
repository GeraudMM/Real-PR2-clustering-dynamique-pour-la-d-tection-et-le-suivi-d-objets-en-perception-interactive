#ifndef RANSAC_SUPERVOXEL_TRACKER_H
#define RANSAC_SUPERVOXEL_TRACKER_H

#include "../papon/supervoxel/sequential_supervoxel_clustering.h"

namespace pcl
{
  class RansacSupervoxelTracker
  {
      // Methods
    public:
      /* \brief Default constructor, default values given to attributes */
      RansacSupervoxelTracker (float threshold = 50.0f, int num_max_iter = 100,
                               float proba_of_pure_inlier = 0.99f,
                               float min_scale = 0.01f,
                               float min_contrast = 0.1f,
                               float search_radius = 0.08f)
      {
        threshold_ = threshold;
        num_max_iter_ = num_max_iter;
        proba_of_pure_inlier_ = proba_of_pure_inlier;
        min_scale_ = min_scale;
        min_contrast_ = min_contrast;
        search_radius_ = search_radius;
      }
      // Getters
      /* \brief Getter for threshold_ attribute */
      float
      getThreshold () const
      { return threshold_; }
      /* \brief Getter for proba_of_pure_inlier_ attribute */
      float
      getProbaOfPureInlier () const
      { return proba_of_pure_inlier_; }
      /* \brief Getter for num_max_iter_ attribute */
      int
      getNumMaxIter () const
      { return num_max_iter_; }
      /* \brief Getter for min_scale_ attribute */
      float
      getMinScale () const
      { return min_scale_; }
      /* \brief Getter for min_contrast_ attribute */
      float
      getMinContrast () const
      { return min_contrast_; }
      /* \brief Getter for search_radius_ attribute */
      float
      getSearchRadius () const
      { return search_radius_; }
      // Setters
      /* \brief Setter for threshold_ attribute */
      void
      setThreshold (float threshold)
      { threshold_= threshold; }
      /* \brief Setter for proba_of_pure_inlier_ attribute */
      void
      setProbaOfPureInlier (float probability)
      { proba_of_pure_inlier_ = probability; }
      /* \brief Setter for num_max_iter_ attribute */
      void
      setNumMaxIter (float num_max_iter)
      { num_max_iter_ = num_max_iter; }
      /* \brief Setter for min_scale_ attribute */
      void
      setMinScale (float min_scale)
      { min_scale_ = min_scale; }
      /* \brief Setter for min_contrast_ attribute */
      void
      setMinContrast (float min_contrast)
      { min_contrast_ = min_contrast; }
      /* \brief Setter for search_radius_ attribute */
      void
      setSearchRadius (float search_radius)
      { search_radius_ = search_radius; }
    private:
      /* \brief Gives the number of samples that need to be drawn in order to
       * have at least one pure inlier sample with given input probability */
      int
      computeRequiredNumberOfIterations ();
      // Attributes
    private:
      // RANSAC algorithm parameters
      float threshold_;
      int num_max_iter_;
      float proba_of_pure_inlier_;
      // SIFT keypoint detection parameters
      float min_scale_;
      float min_contrast_;
      float search_radius_;
  };
}

#endif // RANSAC_SUPERVOXEL_TRACKER_H
