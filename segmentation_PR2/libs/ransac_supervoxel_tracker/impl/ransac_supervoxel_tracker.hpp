#ifndef RANSAC_SUPERVOXEL_TRACKER_HPP
#define RANSAC_SUPERVOXEL_TRACKER_HPP

#include "../ransac_supervoxel_tracker.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::RansacSupervoxelTracker<PointT>::computeIntensityGradientCloud (PointCloudIG::Ptr cloud_ig, const PointCloudI::Ptr xyzi_total_cloud, const NormalCloud::Ptr total_normal_cloud) const
{
  pcl::IntensityGradientEstimation<pcl::PointXYZI, pcl::Normal, pcl::IntensityGradient> gradient_est;
  typename pcl::search::KdTree<pcl::PointXYZI>::Ptr treept2 (new pcl::search::KdTree<pcl::PointXYZI> (false));
  gradient_est.setSearchMethod(treept2);
  gradient_est.setRadiusSearch(seed_resolution_/2.);
  gradient_est.setInputCloud(xyzi_total_cloud);
  gradient_est.setInputNormals(total_normal_cloud);
  gradient_est.compute(*cloud_ig);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::pair< pcl::IndicesPtr, pcl::PointCloud< pcl::Histogram<32> >::Ptr >
pcl::RansacSupervoxelTracker<PointT>::computeRIFTDescriptors (const PointCloudScale sift_result, const PointCloudIG::Ptr cloud_ig, const PointCloudI::Ptr xyzi_total_cloud) const
{
  // Get a pointcloud of keypoints in the type PointXYZI (we are just interested in the XYZ position of those points)
  PointCloudI::Ptr input_keypoints_cloud_i(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_keypoints_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  copyPointCloud(sift_result, *input_keypoints_cloud_rgb);
  PointCloudXYZRGBtoXYZI(*input_keypoints_cloud_rgb, *input_keypoints_cloud_i);

  // Estimate the RIFT feature
  pcl::RIFTEstimation<pcl::PointXYZI, pcl::IntensityGradient, pcl::Histogram<32> > rift_est;
  typename pcl::search::KdTree<pcl::PointXYZI>::Ptr treept3 (new pcl::search::KdTree<pcl::PointXYZI> (false));
  rift_est.setSearchMethod(treept3);
  rift_est.setRadiusSearch(seed_resolution_/2.);
  rift_est.setNrDistanceBins (4);
  rift_est.setNrGradientBins (8);
  // Compute only RIFT descriptor for SIFT keypoints
  rift_est.setInputCloud(input_keypoints_cloud_i);
  rift_est.setInputGradient(cloud_ig);
  // Use all the points in the supervoxel cloud to compute the RIFT descriptor
  rift_est.setSearchSurface(xyzi_total_cloud);

  pcl::IndicesPtr rift_indices = boost::make_shared<std::vector<int>> ();
  rift_indices->reserve (sift_result.size ());

  // We only search the indice of each keypoint in the main PointCloud so we are just interested in their XYZ position
  pcl::search::KdTree<pcl::PointXYZ> point_indices_kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(*xyzi_total_cloud, *xyz_cloud);
  point_indices_kdtree.setInputCloud (xyz_cloud);

  for(auto point_scale: sift_result)
  {
    std::vector<int> indices;
    std::vector<float> sqr_distances;

    pcl::PointXYZ pt(point_scale.x, point_scale.y, point_scale.z);
    point_indices_kdtree.nearestKSearch (pt, 1, indices, sqr_distances);
    int keypoint_indice = indices[0];
    rift_indices->push_back(keypoint_indice);
  }
  // Remove doublons
  sort( rift_indices->begin(), rift_indices->end() );
  rift_indices->erase( unique( rift_indices->begin(), rift_indices->end() ), rift_indices->end() );

  rift_est.setIndices(rift_indices);
  pcl::PointCloud<pcl::Histogram<32>>::Ptr rift_output(new pcl::PointCloud<pcl::Histogram<32>>);
  rift_est.compute (*rift_output);

  return (std::pair<pcl::IndicesPtr, pcl::PointCloud<pcl::Histogram<32>>::Ptr>
          (rift_indices, rift_output));
//  return (KeypointFeatureT (rift_indices, rift_output));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::pair< pcl::IndicesPtr, pcl::PointCloud<pcl::FPFHSignature33>::Ptr >
pcl::RansacSupervoxelTracker<PointT>::computeFPFHDescriptors (const PointCloudScale sift_result, const typename PointCloudT::Ptr cloud, const NormalCloud::Ptr normals) const
{
  IndicesPtr indices (new std::vector<int>);
  // We only search the indice of each keypoint in the main PointCloud so we are just interested in their XYZ position
  pcl::search::KdTree<pcl::PointXYZ> point_indices_kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(*cloud, *xyz_cloud);
  point_indices_kdtree.setInputCloud (xyz_cloud);

  for(auto point_scale: sift_result)
  {
    std::vector<int> n_indices;
    std::vector<float> sqr_distances;

    pcl::PointXYZ pt(point_scale.x, point_scale.y, point_scale.z);
    point_indices_kdtree.nearestKSearch (pt, 1, n_indices, sqr_distances);
    int keypoint_indice = n_indices[0];
    indices->push_back(keypoint_indice);
  }

  // Create the FPFH estimation class, and pass the input dataset+normals to it
  pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud (cloud);
  fpfh.setInputNormals (normals);
  fpfh.setIndices (indices);
  // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

  // Create an empty kdtree representation, and pass it to the FPFH estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

  fpfh.setSearchMethod (tree);

  // Output datasets
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

  // Use all neighbors in a sphere of radius 5cm
  // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
  fpfh.setRadiusSearch (seed_resolution_/2.);

  // Compute the features
  fpfh.compute (*fpfhs);
  return std::pair<IndicesPtr, pcl::PointCloud<pcl::FPFHSignature33>::Ptr> (indices, fpfhs);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::RansacSupervoxelTracker<PointT>::filterKeypoints(const std::pair <pcl::IndicesPtr, PointCloudFeatureT::Ptr > to_filter_keypoints, std::pair <pcl::IndicesPtr, PointCloudFeatureT::Ptr >& filtered_keypoints) const
{
  for (size_t idx = 0; idx < to_filter_keypoints.second->size (); ++idx)
  {
    bool to_remove = false;
//    int nb_of_0s = 0;
    FeatureT descriptor = (*to_filter_keypoints.second)[idx];
    int size_of_descriptor = descriptor.descriptorSize ();
    // 32 = size of descriptor
    for (int i = 0; i < size_of_descriptor; ++i)
    {
      if(std::isnan(descriptor.histogram[i]))//|| descriptor.histogram[i] == 0.0)
      { to_remove = true; break; }
//      if(descriptor.histogram[i] == 0.0)
//      { ++nb_of_0s; }
    }
//    if (nb_of_0s > 2)
//    { to_remove = true; }
    if(!to_remove)
    {
      filtered_keypoints.first->push_back((*to_filter_keypoints.first)[idx]);
      filtered_keypoints.second->push_back(descriptor);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::vector<int>
pcl::RansacSupervoxelTracker<PointT>::computeKeypointsMatches(const std::vector<int> to_match_indices, const PointCloudFeatureT to_match_feature_cloud,
                                                             const std::pair <pcl::IndicesPtr, PointCloudFeatureT::Ptr > indices_point_pair)
{
  // Search the nearest previous keypoint in feature space for each of the maybe inliers
  // in order to compute a transform using singular value decomposition
  // Instantiate search object with 4 randomized trees and 256 checks
  SearchT search (true, CreatorPtrT (new IndexT (4)));
  search.setPointRepresentation (RepresentationPtrT (new DefaultFeatureRepresentation<FeatureT>));
  search.setChecks (128);
//  pcl::search::KdTree<FeatureT> search;
  search.setInputCloud (indices_point_pair.second);
  // Create a vector that contains -1
  std::vector<int> matches_of_to_match(to_match_indices.size (), -1);
  std::vector<int> k_indices;
  std::vector<float> k_sqr_distances;
  // Do search
  std::unordered_map<int, std::pair<std::vector<int>, std::vector<float>>> neighbours_of_inliers;

  for(size_t idx = 0; idx < to_match_feature_cloud.size (); ++idx)
  {
    // We look up for k neighbours with k equal to the number of min
    // inliers
    size_t k = indices_point_pair.first->size ();//to_match_feature_cloud.size ();
    search.nearestKSearch(to_match_feature_cloud[idx], static_cast<int> (k), k_indices, k_sqr_distances);
    neighbours_of_inliers.insert(std::pair<int, std::pair<std::vector<int>, std::vector<float>>>
                                 (idx,
                                  std::pair<std::vector<int>, std::vector<float>>
                                  (k_indices, k_sqr_distances))
                                 );
  }
  // Associate each maybe inlier to its best match in the previous
  // keypoints (priority for order of importance for each neigbour)
  std::vector<int> unmatched_idx(to_match_indices.size ());
  std::iota(unmatched_idx.begin (), unmatched_idx.end (), 1);
  //  for (size_t depth = 0; depth < to_match_indices.size (); ++depth)
  for (size_t depth = 0; depth < indices_point_pair.first->size (); ++depth)
  {
    std::unordered_map<int, std::pair<int, float>> min;
    for (auto idx: unmatched_idx)
    {
      // If the distance equals to 0, it's a bug because we search too far
      // around the point when building neighbours_of_inliers
//      if(neighbours_of_inliers[idx-1].second[depth] != 0.0)
//      {
        std::unordered_map<int, std::pair<int, float>>::iterator map_it = min.find (neighbours_of_inliers[idx-1].first[depth]);
        // Here to handle a bug where sometimes, a keypoints has for neighbour a point with big indice
        // Don't know why it occurs, seems to happen with small input search clouds
        if (neighbours_of_inliers[idx-1].first[depth] > indices_point_pair.first->size ())
          continue;
        // If another point is competing for this match, compare errors
        if (map_it != min.end ())
        {
          if (map_it->second.second > neighbours_of_inliers[idx-1].second[depth])
          {
            map_it->second.first = idx-1;
            map_it->second.second = neighbours_of_inliers[idx-1].second[depth];
          }
        }
        // Otherwise just insert a new element
        else
        {
          min.insert(std::pair<int, std::pair<int, float>>
                     (neighbours_of_inliers[idx-1].first[depth],
                     std::pair<int, float>
                     (idx-1, neighbours_of_inliers[idx-1].second[depth])));
        }
//      }
    }
    // Now allocate each match to each point and remove the allocated
    // point index from unmatched index vector
    for(std::unordered_map<int, std::pair<int, float>>::iterator map_it = min.begin (); map_it != min.end (); ++map_it)
    {
      matches_of_to_match[map_it->second.first] = (*indices_point_pair.first)[map_it->first];
      unmatched_idx.erase (std::remove (unmatched_idx.begin (), unmatched_idx.end (), map_it->second.first+1), unmatched_idx.end ());
    }
  }
  return (matches_of_to_match);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::vector<uint32_t>
pcl::RansacSupervoxelTracker<PointT>::getLabelsOfDynamicSV (SequentialSVMapT &supervoxel_clusters)
{
  std::vector<uint32_t> labels_to_track;
  // Get the labels of the disappeared and fully occluded supervoxels
  int nb_occluded_voxels_by_labels[getMaxLabel ()] = {0};
  int nb_voxels_by_labels[getMaxLabel ()] = {0};
  for(typename LeafVectorT::iterator leaf_itr = sequential_octree_->begin (); leaf_itr != sequential_octree_->end (); ++leaf_itr)
  {
    SequentialVoxelData& voxel = (*leaf_itr)->getData ();
    if(voxel.frame_occluded_ != 0) // It is currently occluded
    { ++nb_occluded_voxels_by_labels[voxel.label_ - 1]; }
    ++nb_voxels_by_labels[voxel.label_ - 1];
  }
  for(auto cluster: supervoxel_clusters)
  {
    uint32_t i = cluster.first;
    // If the sv has disappeared or is fully occluded
    if(nb_voxels_by_labels[i - 1] == 0 ||
       ( (nb_voxels_by_labels[i - 1] == nb_occluded_voxels_by_labels[i - 1]) && (supervoxel_clusters.find(i) != supervoxel_clusters.end()) ) )
    { labels_to_track.push_back(i); }
  }
  return (labels_to_track);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template < typename PointT> void
pcl::RansacSupervoxelTracker<PointT>::initializeSIFT (pcl::SIFTKeypoint<PointT, pcl::PointWithScale> &sift,
                                                     float min_scale, float min_contrast, int n_octaves, int n_scales_per_octave)
{
  typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT> ());
  sift.setSearchMethod(tree);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::RansacSupervoxelTracker<PointT>::computeRIFTRequiredClouds(PointCloudIG::Ptr cloud_ig, PointCloudI::Ptr xyzi_total_cloud,
                                                               typename PointCloudT::Ptr cloud, NormalCloud::Ptr normal_cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzrgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  // Convert previous voxel centroid cloud from XYZRGB/XYZRGBA to XYZI
  copyPointCloud(*cloud, *xyzrgb_cloud);
  PointCloudXYZRGBtoXYZI(*xyzrgb_cloud, *xyzi_total_cloud);

  // Estimate the Intensity Gradient for this cloud
  computeIntensityGradientCloud (cloud_ig, xyzi_total_cloud, normal_cloud);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::RansacSupervoxelTracker<PointT>::computeSIFTKeypointsAndFPFHDescriptors(SequentialSVMapT &supervoxel_clusters,
                                                                            KeypointMapFeatureT &previous_keypoints,
                                                                            int min_nb_of_keypoints,
                                                                            float min_scale, float min_contrast,
                                                                            int n_octaves, int n_scales_per_octave)
{
  // Get the labels that need to be tracked
  std::vector<uint32_t> labels_to_track = getLabelsOfDynamicSV (supervoxel_clusters);
  // Initiliaze the SIFT keypoints structure
  pcl::SIFTKeypoint<PointT, pcl::PointWithScale> sift;
  initializeSIFT (sift, min_scale, min_contrast, n_octaves, n_scales_per_octave);

  // Get SIFT keypoints and RIFT descriptors for each keypoint from precedent cloud for each label
  PointCloudScale sift_result;
  for(const auto& label: labels_to_track)
  {
    if(supervoxel_clusters[label]->voxels_->size () > 0)// Need to get a good metric
    {
      // Estimate the sift interest points using Intensity values from RGB values
      sift_result.clear ();
      sift.setInputCloud(supervoxel_clusters[label]->voxels_);
      sift.compute(sift_result);
      if(sift_result.size () > min_nb_of_keypoints)
      {
        // Compute the RIFT descriptors
        KeypointFeatureT fpfh_output = computeFPFHDescriptors (sift_result, prev_voxel_centroid_cloud_, prev_voxel_centroid_normal_cloud_);
        KeypointFeatureT filtered_fpfh_output
            (boost::make_shared<std::vector<int>> (), boost::make_shared<PointCloudFeatureT> ());
        filterKeypoints(fpfh_output, filtered_fpfh_output);
        if(filtered_fpfh_output.second->size () > min_nb_of_keypoints)// Min number of keypoints in order to track the supervoxel
        {
          previous_keypoints.insert (std::pair<uint32_t, KeypointFeatureT> (label, filtered_fpfh_output));
        }
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::RansacSupervoxelTracker<PointT>::computeSIFTKeypointsAndFPFHDescriptors(KeypointFeatureT &current_keypoints,
                                                                            int min_nb_of_keypoints,
                                                                            float min_scale, float min_contrast,
                                                                            int n_octaves, int n_scales_per_octave)
{
  // Initiliaze the SIFT keypoints structure
  pcl::SIFTKeypoint<PointT, pcl::PointWithScale> sift;
  initializeSIFT (sift, min_scale, min_contrast, n_octaves, n_scales_per_octave);
  // Get SIFT keypoints and RIFT descriptors for each keypoint from precedent cloud for each label
  PointCloudScale sift_result;
  // Estimate the sift interest points using Intensity values from RGB values
  sift_result.clear ();
  sift.setInputCloud(unlabeled_voxel_centroid_cloud_);
  sift.compute(sift_result);
  if(sift_result.size () > min_nb_of_keypoints)
  {
    // Compute the RIFT descriptors
    KeypointFeatureT fpfh_output = computeFPFHDescriptors (sift_result, voxel_centroid_cloud_, getVoxelNormalCloud ());
    KeypointFeatureT filtered_fpfh_output
        (boost::make_shared<std::vector<int>> (), boost::make_shared<PointCloudFeatureT> ());
    filterKeypoints(fpfh_output, filtered_fpfh_output);
    if(filtered_fpfh_output.second->size () > min_nb_of_keypoints)// Min number of keypoints in order to track the supervoxel
    {
      current_keypoints = filtered_fpfh_output;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//template <typename PointT> void
//pcl::RansacSupervoxelTracker<PointT>::computeSIFTKeypointsAndRIFTDescriptors(SequentialSVMapT &supervoxel_clusters,
//                                                                            KeypointMapFeatureT &previous_keypoints,
//                                                                            int min_nb_of_keypoints,
//                                                                            float min_scale, float min_contrast,
//                                                                            int n_octaves, int n_scales_per_octave)
//{
//  // Get the labels that need to be tracked
//  std::vector<uint32_t> labels_to_track = getLabelsOfDynamicSV (supervoxel_clusters);
//  // Initiliaze the SIFT keypoints structure
//  pcl::SIFTKeypoint<PointT, pcl::PointWithScale> sift;
//  initializeSIFT (sift, min_scale, min_contrast, n_octaves, n_scales_per_octave);
//  PointCloudScale sift_result;
//  // Compute clouds
//  PointCloudI::Ptr xyzi_total_cloud (new PointCloudI);
//  PointCloudIG::Ptr cloud_ig (new PointCloudIG);
//  computeRIFTRequiredClouds(cloud_ig, xyzi_total_cloud,
//                            prev_voxel_centroid_cloud_, prev_voxel_centroid_normal_cloud_);
//  // Get SIFT keypoints and RIFT descriptors for each keypoint from precedent cloud for each label
//  for(auto label: labels_to_track)
//  {
//    if(supervoxel_clusters[label]->voxels_->size () > 20)// Need to get a good metric
//    {
//      // Estimate the sift interest points using Intensity values from RGB values
//      sift_result.clear ();
//      sift.setInputCloud(supervoxel_clusters[label]->voxels_);
//      sift.compute(sift_result);
//      if(sift_result.size () > min_nb_of_keypoints)
//      {
//        // Compute the RIFT descriptors
//        KeypointFeatureT rift_output = computeRIFTDescriptors (sift_result, cloud_ig, xyzi_total_cloud);
//        KeypointFeatureT filtered_rift_output
//            (boost::make_shared<std::vector<int>> (), boost::make_shared<PointCloudFeatureT> ());
//        filterKeypoints(rift_output, filtered_rift_output);
//        if(filtered_rift_output.second->size () > min_nb_of_keypoints)// Min number of keypoints in order to track the supervoxel
//        {
//          previous_keypoints.insert (std::pair<uint32_t, KeypointFeatureT> (label, filtered_rift_output));
//        }
//      }
//    }
//  }
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//template <typename PointT> void
//pcl::RansacSupervoxelTracker<PointT>::computeSIFTKeypointsAndRIFTDescriptors(KeypointFeatureT &current_keypoints,
//                                                                            int min_nb_of_keypoints,
//                                                                            float min_scale, float min_contrast,
//                                                                            int n_octaves, int n_scales_per_octave)
//{
//  // Initiliaze the SIFT keypoints structure
//  pcl::SIFTKeypoint<PointT, pcl::PointWithScale> sift;
//  initializeSIFT (sift, min_scale, min_contrast, n_octaves, n_scales_per_octave);
//  PointCloudScale sift_result;
//  // Compute clouds
//  PointCloudI::Ptr xyzi_total_cloud (new PointCloudI);
//  PointCloudIG::Ptr cloud_ig (new PointCloudIG);
//  //  computeRIFTRequiredClouds(cloud_ig, xyzi_total_cloud,
//  //                            getUnlabeledVoxelCentroidCloud (), getUnlabeledVoxelNormalCloud ());
//  computeRIFTRequiredClouds(cloud_ig, xyzi_total_cloud,
//                            unlabeled_voxel_centroid_cloud_, unlabeled_voxel_centroid_normal_cloud_);
//  // Estimate the sift interest points using Intensity values from RGB values
//  //  sift.setInputCloud(getUnlabeledVoxelCentroidCloud ());
//  sift.setInputCloud(unlabeled_voxel_centroid_cloud_);
//  sift.compute(sift_result);
//  if(sift_result.size () > min_nb_of_keypoints)
//  {
//    // Compute the RIFT descriptors
//    current_keypoints = computeRIFTDescriptors (sift_result, cloud_ig, xyzi_total_cloud);
//    // Filter the keypoints
//    std::pair <pcl::IndicesPtr, PointCloudFeatureT::Ptr > filtered_keypoints
//        (boost::make_shared<std::vector<int>> (), boost::make_shared<PointCloudFeatureT> ());
//    filterKeypoints(current_keypoints, filtered_keypoints);
//    if(filtered_keypoints.second->size () > min_nb_of_keypoints)// Min number of keypoints in order to track the supervoxel
//    {
//      current_keypoints = filtered_keypoints;
//    }
//  }
//}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::RansacSupervoxelTracker<PointT>::samplePotentialInliers(std::vector<int> &indices,
                                                            PointCloudFeatureT::Ptr &cloud,
                                                            std::vector<int> &potential_inliers,
                                                            PointCloudFeatureT::Ptr &potential_inliers_feature_cloud,
                                                            int nb_to_sample)
{
  // Select min_number_of_inliers values from data (the SIFT keypoints from current frame)
  for (int j = 0; j < nb_to_sample; ++j)
  {
    // True random generator
    std::random_device generator;
    std::uniform_int_distribution<int> distribution(0, indices.size () - 1);
    int rand_indice = distribution (generator);
    potential_inliers.push_back (indices[rand_indice]);
    potential_inliers_feature_cloud->push_back (cloud->at(rand_indice));
    indices.erase(indices.begin () + rand_indice);
    cloud->erase (cloud->begin () + rand_indice);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> Eigen::Matrix<float, 4, 4>
pcl::RansacSupervoxelTracker<PointT>::computeRigidTransformation(std::vector<int> prev_indices, std::vector<int> curr_indices)
{
  // Compute the transform between the points sampled from data and the previous keypoints
  boost::shared_ptr< pcl::registration::TransformationEstimation< PointT, PointT > > estimator; // Generic container for estimators
  estimator.reset ( new pcl::registration::TransformationEstimationSVD < PointT, PointT > () ); // SVD transform estimator
  Eigen::Matrix<float, 4, 4> transformation_est;
  typename PointCloudT::Ptr cloud_src, cloud_tgt;
  cloud_src.reset(new PointCloudT);
  cloud_tgt.reset(new PointCloudT);
  for (size_t idx = 0; idx < curr_indices.size (); ++idx)
  {
    // Handle bug
    if(prev_indices[idx] < prev_voxel_centroid_cloud_->size ())
    {
      cloud_src->push_back ((*prev_voxel_centroid_cloud_)[prev_indices[idx]]);
      //      cloud_tgt->push_back ((*getUnlabeledVoxelCentroidCloud ())[curr_indices[idx]]);
      cloud_tgt->push_back ((*unlabeled_voxel_centroid_cloud_)[curr_indices[idx]]);
    }
  }
  estimator->estimateRigidTransformation (*cloud_src, *cloud_tgt, transformation_est);
  return (transformation_est);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::RansacSupervoxelTracker<PointT>::findInliers(const PointCloudFeatureT::Ptr &search_cloud,
                                                 const std::vector<int> &spatial_neighbors,
                                                 const std::vector<int> &all_indices,
                                                 const PointCloudFeatureT::Ptr &all_cloud,
                                                 std::vector<int> &potential_inliers,
                                                 PointCloudFeatureT::Ptr &potential_inliers_cloud,
                                                 float threshold, float* err)
{
  *err = 0.0f;
  bool no_inliers = true;
  // Search the nearest previous keypoint in feature space for each of the maybe inliers
  // in order to compute a transform using singular value decomposition
  // Instantiate search object with 4 randomized trees and 256 checks
  SearchT search (true, CreatorPtrT (new IndexT (4)));
  search.setPointRepresentation (RepresentationPtrT (new DefaultFeatureRepresentation<FeatureT>));
  search.setChecks (128);
  search.setInputCloud (search_cloud);
  for(auto neigh_ind: spatial_neighbors)
  {
    std::vector<int>::const_iterator vec_it = std::find (all_indices.begin (), all_indices.end (), neigh_ind);
    // If indice in the current keypoints (without maybe inliers)
    if( vec_it != all_indices.end ())
    {
      std::vector<int> indices;
      std::vector<float> distances;
      // Search for the nearest neighbour in feature space
      // between corresponding keypoint and keypoints in previous
      // supervoxel
      search.nearestKSearch((*all_cloud)[vec_it-all_indices.begin ()], 1, indices, distances);
      // If the NN of this point is close enough it's an inlier
//      std::cout << "idx: " << vec_it-all_indices.begin () << " distance min: " << distances[0] << "\n";
      if(distances[0] < threshold && distances[0] != 0.0)
      {
        *err += distances[0];
        no_inliers = false;
        potential_inliers.push_back (neigh_ind);
        potential_inliers_cloud->push_back (all_cloud->at(vec_it-all_indices.begin ()));
      }
    }
  }
  if(no_inliers)
  { *err = std::numeric_limits<float>::max (); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::unordered_map<uint32_t, Eigen::Matrix<float, 4, 4>>
pcl::RansacSupervoxelTracker<PointT>::getMatchesRANSAC (SequentialSVMapT &supervoxel_clusters)
{
  // This will store the transforms that are valid
  std::unordered_map<uint32_t, Eigen::Matrix<float, 4, 4>> found_transforms;
  // previous_keypoints is a map where the key is the label of the concerned supervoxel, and the value is a pair
  // consisting as first element of the Indices of the keypoints in the previous voxel centroid cloud and as second
  // element of the RIFT descriptor matching this indice.
  KeypointMapFeatureT previous_keypoints;
  KeypointFeatureT current_keypoints;
  // Parameters for sift computation
  float min_scale = 0.005f;
  float min_contrast = 0.6f;
  // RANSAC variables
  int min_number_of_inliers = 3;
  float proba_of_pure_inlier = 0.99f;
  int num_max_iter = 100; // Same as in Van Hoof paper
  float threshold = 50.f;
  std::vector<uint32_t> labels;

  if(getMaxLabel() > 0)
  {
    ///////////////////////////////////////////////////////////////////////////////////////////
    /////////////////COMPUTE KEYPOINTS AND DESCRIPTORS FOR PREVIOUS CLOUD//////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
//    computeSIFTKeypointsAndRIFTDescriptors(supervoxel_clusters, previous_keypoints,
//                                           min_number_of_inliers, min_scale, min_contrast);
    computeSIFTKeypointsAndFPFHDescriptors(supervoxel_clusters, previous_keypoints,
                                           min_number_of_inliers, min_scale, min_contrast);
    if(previous_keypoints.size () > 0)
    {
      ///////////////////////////////////////////////////////////////////////////////////////////
      //////////////////COMPUTE KEYPOINTS AND DESCRIPTORS FOR CURRENT CLOUD//////////////////////
      ///////////////////////////////////////////////////////////////////////////////////////////
//      computeSIFTKeypointsAndRIFTDescriptors(current_keypoints, min_number_of_inliers,min_scale, min_contrast);
      computeSIFTKeypointsAndFPFHDescriptors(current_keypoints, min_number_of_inliers,min_scale, min_contrast);
      // If there is not enough keypoints in current scene
      if (current_keypoints.first->size () < min_number_of_inliers)
      { return (found_transforms); }
      current_keypoints_indices_ = *(current_keypoints.first);
      previous_keypoints_indices_.clear();
      for (auto pair: previous_keypoints)
      {
        previous_keypoints_indices_.insert(std::end(previous_keypoints_indices_),
                                           std::begin(*pair.second.first), std::end(*pair.second.first));
      }
      for(auto pair: previous_keypoints)
      {
        uint32_t label = pair.first;
        Eigen::Matrix<float, 4, 4> best_fit;
        int max_nb_of_inliers = 0;
        float best_err = std::numeric_limits<float>::max ();

        // RANSAC implementation to find the transform that best explain
        // new keypoints observations
        for(size_t i = 0; i < num_max_iter; ++i)
        {
          // Temporary structures that store copies from current keypoints data
//          std::vector<int> tmp_indices = *current_keypoints.first;
//          PointCloudFeatureT::Ptr tmp_cloud (new PointCloudFeatureT);
//          copyPointCloud (*current_keypoints.second, *tmp_cloud);
//          // Temporary structures that store copies from previous keypoints data
//          std::vector<int> tmp_indices_prev = *pair.second.first;
//          PointCloudFeatureT::Ptr tmp_cloud_prev (new PointCloudFeatureT);
//          copyPointCloud (*pair.second.second, *tmp_cloud_prev);
//          // Structures that store potential inliers data
//          std::vector<int> maybe_inliers;
//          maybe_inliers.reserve (min_number_of_inliers);
//          PointCloudFeatureT::Ptr maybe_inliers_feature_cloud (new PointCloud<FeatureT>);
//          // Draw min_number_of_inliers samples from the observed keypoint cloud
//          //          samplePotentialInliers(tmp_indices, tmp_cloud, maybe_inliers, maybe_inliers_feature_cloud, min_number_of_inliers);
//          samplePotentialInliers(tmp_indices_prev, tmp_cloud_prev, maybe_inliers, maybe_inliers_feature_cloud, min_number_of_inliers);
//          // Compute keypoints matches with maybe inliers
//          std::vector<int> matches_of_maybe_inliers = computeKeypointsMatches(maybe_inliers, *maybe_inliers_feature_cloud,
//                                                                              current_keypoints);
//          // Compute the transform between the points sampled from data and the previous keypoints
//          Eigen::Matrix<float, 4, 4> transformation_est = computeRigidTransformation(maybe_inliers, matches_of_maybe_inliers);
//          // Compute the new transformed centroid of the current supervoxel being matched
//          pcl::PointXYZRGBA prev_centroid_tmp = supervoxel_clusters[label]->centroid_;
//          Eigen::Vector4f new_centroid, prev_centroid(prev_centroid_tmp.x, prev_centroid_tmp.y, prev_centroid_tmp.z, 1);
//          new_centroid = transformation_est*prev_centroid;
//          pcl::PointXYZ cent(new_centroid[0], new_centroid[1], new_centroid[2]);
          // Temporary structures that store copies from current keypoints data
          std::vector<int> tmp_indices = *current_keypoints.first;
          PointCloudFeatureT::Ptr tmp_cloud (new PointCloudFeatureT);
          copyPointCloud (*current_keypoints.second, *tmp_cloud);
          // Structures that store potential inliers data
          std::vector<int> maybe_inliers;
          maybe_inliers.reserve (min_number_of_inliers);
          PointCloudFeatureT::Ptr maybe_inliers_feature_cloud (new PointCloud<FeatureT>);
          // Draw min_number_of_inliers samples from the observed keypoint cloud
          samplePotentialInliers(tmp_indices, tmp_cloud, maybe_inliers, maybe_inliers_feature_cloud, min_number_of_inliers);
          // Compute keypoints matches with maybe inliers
          std::vector<int> matches_of_maybe_inliers = computeKeypointsMatches(maybe_inliers, *maybe_inliers_feature_cloud,
                                                                              pair.second);
          // Compute the transform between the points sampled from data and the previous keypoints
          Eigen::Matrix<float, 4, 4> transformation_est = computeRigidTransformation(matches_of_maybe_inliers, maybe_inliers);
          // Compute the new transformed centroid of the current supervoxel being matched
          pcl::PointXYZRGBA prev_centroid_tmp = supervoxel_clusters[label]->centroid_;
          Eigen::Vector4f new_centroid, prev_centroid(prev_centroid_tmp.x, prev_centroid_tmp.y, prev_centroid_tmp.z, 1);
          new_centroid = transformation_est*prev_centroid;
          pcl::PointXYZ cent(new_centroid[0], new_centroid[1], new_centroid[2]);

          // Do radius search around new estimated centroid
          // This search is only based on spatiality in a radius of
          // seed_resolution (we are looking for the previous supervoxel)
          std::vector<int> k_indices;
          std::vector<float> k_sqr_distances;
//          typename pcl::search::KdTree<pcl::PointXYZ>::Ptr unlabeled_voxel_cloud_search;
//          unlabeled_voxel_cloud_search.reset (new pcl::search::KdTree<pcl::PointXYZ>);
//          pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//          //          copyPointCloud(*getUnlabeledVoxelCentroidCloud (), *xyz_cloud);
//          copyPointCloud(*unlabeled_voxel_centroid_cloud_, *xyz_cloud);
//          unlabeled_voxel_cloud_search->setInputCloud (xyz_cloud);
//          unlabeled_voxel_cloud_search->radiusSearch (cent, seed_resolution_/2., k_indices, k_sqr_distances);
          typename pcl::search::KdTree<pcl::PointXYZ>::Ptr voxel_cloud_search;
          voxel_cloud_search.reset (new pcl::search::KdTree<pcl::PointXYZ>);
          pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
          copyPointCloud(*voxel_centroid_cloud_, *xyz_cloud);
          voxel_cloud_search->setInputCloud (xyz_cloud);
          voxel_cloud_search->radiusSearch (cent, seed_resolution_/2., k_indices, k_sqr_distances);
          float err;
          // Find the current keypoints below thresh that are in this potential supervoxel
          findInliers(pair.second.second, k_indices, tmp_indices, tmp_cloud, maybe_inliers,
                      maybe_inliers_feature_cloud, threshold, &err);
          // Condition: there is more than a quarter of possible matches
          // with a minimum of 6 (like in Van Hoof's paper)
          if(maybe_inliers.size () >= pair.second.second->size ()/4)
            //             && maybe_inliers.size () > 6)
          {
            // Compute keypoints matches with maybe inliers
            std::vector<int> matches_of_maybe_inliers = computeKeypointsMatches(maybe_inliers, *maybe_inliers_feature_cloud,
                                                                                pair.second);
            transformation_est = computeRigidTransformation(matches_of_maybe_inliers, maybe_inliers);
            // Compute the new transformed centroid of the current supervoxel being matched
            new_centroid = transformation_est*prev_centroid;
            pcl::PointXYZ cent_refit(new_centroid[0], new_centroid[1], new_centroid[2]);
            // Do radius search around new estimated centroid
            // This search is only based on spatiality in a radius of
            // seed_resolution (we are looking for the previous supervoxel)
//            unlabeled_voxel_cloud_search->radiusSearch (cent_refit, seed_resolution_/2., k_indices, k_sqr_distances);
            voxel_cloud_search->radiusSearch (cent_refit, seed_resolution_/2., k_indices, k_sqr_distances);
            // Find the current keypoints below thresh that are in this potential supervoxel
            std::vector<int> inliers;
            PointCloudFeatureT::Ptr inliers_feature_cloud (new PointCloudFeatureT);
            findInliers(pair.second.second, k_indices, *current_keypoints.first,
                        current_keypoints.second, inliers, inliers_feature_cloud, threshold, &err);
            // score = coeff*scoreinliers + coeff*scoreerr
            // scoreinliers = 1
            // scoreerr = 1
            float coeff_in = 1.f; float coeff_err = 3.f;
            float best_score = coeff_in * max_nb_of_inliers/pair.second.second->size () + ((coeff_err * (threshold - best_err) > 0)?coeff_err * (threshold - best_err)/threshold:0);
            float curr_score = coeff_in * inliers.size()/pair.second.second->size () + ((coeff_err * (threshold - err) > 0)?coeff_err * (threshold - err)/threshold:0);
            //            std::cout << "err: " << err << " curr: " << curr_score << " best: " << best_score << "\n";
            //            if(inliers.size () > maybe_inliers.size ()
            //               && max_nb_of_inliers <= inliers.size ()
            //               && err <= best_err)
            if (curr_score > best_score
                && inliers.size () >= maybe_inliers.size ())
            {
              best_fit = transformation_est;
              max_nb_of_inliers = inliers.size ();
              best_err = err;
            }
          }
        }
        std::cout << "SV" << label << " best err: " << best_err << "\n";
        if(max_nb_of_inliers>0)
        {
          std::cout << "SV w/ label " << label << " was matched !\nThe found transform has "
                    << max_nb_of_inliers << " inliers\n";
          found_transforms.insert (std::pair<uint32_t, Eigen::Matrix<float, 4, 4>>
                                   (label, best_fit));
          labels.push_back(label);
        }
      }
    }
  }
  // Update the octree to remove the voxels of the matched SVs
  sequential_octree_->updateOctreeFromMatchedClouds(labels);
  computeVoxelData (); // COULD DO BETTER DON'T NEED TO RECALCULATE THE NORMALS
  return (found_transforms);
}

#endif // RANSAC_SUPERVOXEL_TRACKER_HPP
