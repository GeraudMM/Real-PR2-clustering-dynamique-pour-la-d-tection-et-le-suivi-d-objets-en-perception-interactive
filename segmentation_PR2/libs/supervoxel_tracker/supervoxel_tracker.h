#ifndef SUPERVOXEL_TRACKER_H_
#define SUPERVOXEL_TRACKER_H_

// STL Includes
#include <map>
#include <utility>
// PCL Common Includes
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
// PCL Tracker Includes
#include <pcl/tracking/tracker.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/normal_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/passthrough.h>
// Local libairies Includes
#include "../papon/supervoxel/sequential_supervoxel_clustering.h"
#include "label_coherence.h"
// TBB Library
#include <tbb/tbb.h>
#include "tbb/concurrent_hash_map.h"

namespace pcl
{
  template <typename PointT, typename StateT=pcl::tracking::ParticleXYZRPY>
  class SupervoxelTracker
  {
    public:
      typedef pcl::PointXYZRGBNormal PointNormal;
      typedef pcl::PointCloud<PointT> PointCloudT;
      typedef typename PointCloudT::Ptr PointCloudPtrT;
      typedef typename PointCloudT::ConstPtr PointCloudConstPtrT;
      typedef pcl::tracking::ParticleFilterTracker< PointT, StateT> TrackerT;
//      typedef std::map< uint32_t, TrackerT* > TrackerMapT;
      typedef tbb::concurrent_hash_map< uint32_t, TrackerT* > TrackerMapT;

      typedef typename pcl::SequentialSV<PointT>::Ptr SequentialSVPtr;
      typedef std::map< uint32_t, SequentialSVPtr> SequentialSVMap;

      typedef std::map< uint32_t, StateT > StateMap;
    private:
      TrackerMapT trackers_;

    public:
      /** \brief Default constructor
        */
      SupervoxelTracker ()
      {

      }
      /** \brief Constructor that initialize the trackers with supervoxels clusters
        */
      SupervoxelTracker (const SequentialSVMap supervoxel_clusters)
      {
        setReferenceClouds(supervoxel_clusters);
      }
      /** \brief This method is used to replace all the trackers to follow new reference supervoxels
        * \note Should be use wether at initialisation or to reset the trackers with new reference supervoxels
        */
      void
      setReferenceClouds (const SequentialSVMap supervoxel_clusters);
      /** \brief This method is used to add/replace the reference cloud of a supervoxel label
        * \note this method creates a new tracker for the concerned label
        */
      void
      addReferenceCloud(const uint32_t label, PointCloudPtrT target_cloud);

      /** \brief This method is used to add/replace the reference cloud of a supervoxel label
        * \note this method creates a new tracker for the concerned label
        */
      void
      addReferenceCloud(const uint32_t label, SequentialSVPtr supervoxel_cluster);

      /** \brief This method is used to delete a tracker of a specified supervoxel by label
        */
      void
      deleteTrackerAt(uint32_t label);

      /** \brief return the tracker with key label
        */
      TrackerT*
      getTrackerAt(uint32_t label);
      /** \brief This method is used to get all the predicted states from the particle filters for each supervoxel
        */
      std::map< uint32_t, StateT>
      track(const PointCloudConstPtrT cloud);


    private:
      void
      gridSampleApprox (const PointCloudConstPtrT &cloud, PointCloudT &result, double leaf_size);
      //Filter along a specified dimension
      void
      filterPassThrough (const PointCloudConstPtrT& cloud, PointCloudT& result);
      /** \brief Run the compute of each tracker using TBB to parallelize the calculations */
      void
      parallelTrack (StateMap* states, PointCloudPtrT cloud_pass_downsampled);
      /** \brief Used to setup the coherences of each tracker. See supervoxel_tracker.cpp for specializations */
      void
      setupCoherences (typename pcl::tracking::PointCloudCoherence<PointT>::Ptr coherence);
  };
}

#ifdef PCL_NO_PRECOMPILE
#include "impl/supervoxel_tracker.hpp"
#endif

#endif // SUPERVOXEL_TRACKER_H_
