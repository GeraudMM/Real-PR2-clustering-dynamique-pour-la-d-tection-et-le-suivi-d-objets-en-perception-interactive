#ifndef SUPERVOXEL_TRACKER_HPP_
#define SUPERVOXEL_TRACKER_HPP_

#include "../supervoxel_tracker.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> void
pcl::SupervoxelTracker<PointT, StateT>::setReferenceClouds (SequentialSVMap supervoxel_clusters)
{
  // Erase all previous trackers
//  trackers_.clear ();
  int i=0;
  typename TrackerMapT::accessor a;
  // Iterate over all supervoxel clusters
  typename pcl::SequentialSVClustering<PointT>::SequentialSVMapT::iterator sv_itr;
  for(sv_itr = supervoxel_clusters.begin (); sv_itr != supervoxel_clusters.end (); ++sv_itr)
  {
    uint32_t label = sv_itr->first;
    if(sv_itr->second->isNew () || !(trackers_.find (a, label)))
    {
      i++;
      // Remove the previous tracker on this label if there was one
      trackers_.erase (label);
      // Push the new reference cloud for this label
      PointCloudPtrT target_cloud (new PointCloudT);
      pcl::copyPointCloud (*(sv_itr->second->voxels_), *target_cloud);
      PointT centroid;
      pcl::copyPoint (sv_itr->second->centroid_, centroid);
      addReferenceCloud(label, target_cloud);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> void
pcl::SupervoxelTracker<PointT, StateT>::addReferenceCloud(uint32_t label, SequentialSVPtr supervoxel_cluster)
{
  PointCloudPtrT target_cloud(new PointCloudT);
  pcl::copyPointCloud (*(supervoxel_cluster->voxels_), *target_cloud);
  addReferenceCloud(label, target_cloud);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> void
pcl::SupervoxelTracker<PointT, StateT>::addReferenceCloud(uint32_t label, PointCloudPtrT target_cloud)
{
  pcl::tracking::KLDAdaptiveParticleFilterOMPTracker< PointT, StateT >* tracker = new pcl::tracking::KLDAdaptiveParticleFilterOMPTracker< PointT, StateT > ();

  // Set Parameters
  double downsampling_grid_size = 0.002;
  std::vector<double> default_step_covariance = std::vector<double> (6, 0.015 * 0.015);
  default_step_covariance[3] *= 40.0;
  default_step_covariance[4] *= 40.0;
  default_step_covariance[5] *= 40.0;

  std::vector<double> initial_noise_covariance = std::vector<double> (6, 0.00001);
  std::vector<double> default_initial_mean = std::vector<double> (6, 0.0);

  StateT bin_size;
  bin_size.x = 0.1f;
  bin_size.y = 0.1f;
  bin_size.z = 0.1f;
  bin_size.roll = 0.1f;
  bin_size.pitch = 0.1f;
  bin_size.yaw = 0.1f;


  // Set all parameters for  KLDAdaptiveParticleFilterOMPTracker
  tracker->setMaximumParticleNum (1000);
  tracker->setDelta (0.99);
  tracker->setEpsilon (0.2);
  tracker->setBinSize (bin_size);

  // Set all parameters for  ParticleFilter
  tracker->setTrans (Eigen::Affine3f::Identity ());
  tracker->setStepNoiseCovariance (default_step_covariance);
  tracker->setInitialNoiseCovariance (initial_noise_covariance);
  tracker->setInitialNoiseMean (default_initial_mean);
  tracker->setIterationNum (1);
  tracker->setParticleNum (600);
  tracker->setResampleLikelihoodThr(0.00);
  tracker->setUseNormal (false);


  // Setup coherence object for tracking
  typename pcl::tracking::ApproxNearestPairPointCloudCoherence<PointT>::Ptr coherence (new pcl::tracking::ApproxNearestPairPointCloudCoherence<PointT>);

  setupCoherences(coherence);

  typename pcl::search::Octree<PointT>::Ptr search (new pcl::search::Octree<PointT> (0.01));
  coherence->setSearchMethod (search);
  coherence->setMaximumDistance (0.01);

  tracker->setCloudCoherence (coherence);

  // Prepare the model of tracker's target
  Eigen::Vector4f c;
  Eigen::Affine3f trans = Eigen::Affine3f::Identity ();
  PointCloudPtrT transed_ref (new PointCloudT);
  PointCloudPtrT transed_ref_downsampled (new PointCloudT);

  pcl::compute3DCentroid<PointT> (*target_cloud, c);
  trans.translation ().matrix () = Eigen::Vector3f (c[0], c[1], c[2]);
  pcl::transformPointCloud<PointT> (*target_cloud, *transed_ref, trans.inverse());
  gridSampleApprox (transed_ref, *transed_ref_downsampled, downsampling_grid_size);

  // Set reference model and trans
  tracker->setReferenceCloud (transed_ref_downsampled);
  tracker->setTrans (trans);

  trackers_.insert ( std::make_pair(label, tracker));//std::pair< uint32_t, pcl::tracking::KLDAdaptiveParticleFilterOMPTracker< PointT, StateT > > (label, tracker));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> void
pcl::SupervoxelTracker<PointT, StateT>::deleteTrackerAt(uint32_t label)
{
  // Erase Tracker by key, which is label
  trackers_.erase(label);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> pcl::tracking::ParticleFilterTracker< PointT, StateT>*
pcl::SupervoxelTracker<PointT, StateT>::getTrackerAt(uint32_t label)
{
  typename TrackerMapT::accessor a;
  trackers_.find(a, label);
  return a->second;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> std::map< uint32_t, StateT>
pcl::SupervoxelTracker<PointT, StateT>::track(PointCloudConstPtrT cloud)
{
  double downsampling_grid_size = 0.002;
  PointCloudPtrT cloud_pass(new PointCloudT);
  PointCloudPtrT cloud_pass_downsampled(new PointCloudT);
  filterPassThrough (cloud, *cloud_pass);
  gridSampleApprox (cloud_pass, *cloud_pass_downsampled, downsampling_grid_size);

  StateMap states;
  parallelTrack(&states, cloud_pass);
  //    typename TrackerMapT::iterator itr;
  //    for (itr = trackers_.begin(); itr != trackers_.end(); ++itr)
  //    {
  //      // Get the label of the current tracker being processed
  //      uint32_t label = itr->first;
  //      // Compute the new states
  //      itr->second->setInputCloud(cloud_pass_downsampled);
  //      itr->second->compute();
  //      // Add the predicted state to the return
  //      states.insert (std::make_pair(label, itr->second->getResult ()));
  //    }
  return states;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> void
pcl::SupervoxelTracker<PointT, StateT>::parallelTrack (StateMap* states, PointCloudPtrT cloud_pass_downsampled)
{
  tbb::parallel_for(trackers_.range (),
                    [=](const typename TrackerMapT::range_type& r)
  {
    for (typename TrackerMapT::iterator itr = r.begin () ; itr != r.end (); ++itr)
    {
      // Get the label of the current tracker being processed
      uint32_t label = itr->first;
      // Compute the new states
      itr->second->setInputCloud(cloud_pass_downsampled);
      itr->second->compute();
      // Add the predicted state to the return
      states->insert (std::make_pair(label, itr->second->getResult ()));
    }
  }
  );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> void
pcl::SupervoxelTracker<PointT, StateT>::setupCoherences (typename pcl::tracking::PointCloudCoherence<PointT>::Ptr coherence)
{
  // Default use at least distance coherence
  typename pcl::tracking::DistanceCoherence<PointT>::Ptr distance_coherence (new pcl::tracking::DistanceCoherence<PointT>);
  coherence->addPointCoherence (distance_coherence);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> void
pcl::SupervoxelTracker<PointT, StateT>::gridSampleApprox (const PointCloudConstPtrT &cloud, PointCloudT &result, double leaf_size)
{
  pcl::ApproximateVoxelGrid<PointT> grid;
  grid.setLeafSize (static_cast<float> (leaf_size), static_cast<float> (leaf_size), static_cast<float> (leaf_size));
  grid.setInputCloud (cloud);
  grid.filter (result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename StateT> void
pcl::SupervoxelTracker<PointT, StateT>::filterPassThrough (const PointCloudConstPtrT& cloud, PointCloudT& result)
{
  pcl::PassThrough<PointT> pass;
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 10.0);
  pass.setKeepOrganized (false);
  pass.setInputCloud (cloud);
  pass.filter (result);
}

#define PCL_INSTANTIATE_SupervoxelTracker(T, ST) template class PCL_EXPORTS pcl::SupervoxelTracker<T, ST>;

#endif // SUPERVOXEL_TRACKER_HPP_
