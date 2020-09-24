
/*
  * Software License Agreement (BSD License)
  *
  *  Point Cloud Library (PCL) - www.pointclouds.org
  *
  *  All rights reserved.
  *
  *  Redistribution and use in source and binary forms, with or without
  *  modification, are permitted provided that the following conditions
  *  are met:
  *
  *   * Redistributions of source code must retain the above copyright
  *     notice, this list of conditions and the following disclaimer.
  *   * Redistributions in binary form must reproduce the above
  *     copyright notice, this list of conditions and the following
  *     disclaimer in the documentation and/or other materials provided
  *     with the distribution.
  *   * Neither the name of Willow Garage, Inc. nor the names of its
  *     contributors may be used to endorse or promote products derived
  *     from this software without specific prior written permission.
  *
  *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  *  POSSIBILITY OF SUCH DAMAGE.
  *
  * Author : Elias Hanna
  * Email  : h.elias@hotmail.fr
  *
  */

#ifndef PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_H_
#define PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_H_

#include <pcl/segmentation/supervoxel_clustering.h>
#include "../octree/octree_pointcloud_sequential.h"
#include <tbb/tbb.h>
#include <pcl/recognition/ransac_based/obj_rec_ransac.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/rift.h>
#include <pcl/features/intensity_gradient.h>
#include <pcl/point_types_conversion.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/flann_search.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <boost/thread/mutex.hpp>
#include <pcl/filters/radius_outlier_removal.h>

namespace pcl
{
  struct FPFHCIELabSignature36
  {
      float histogram[36];
      static int descriptorSize () { return 36; }
  };
}
POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::FPFHCIELabSignature36,
                                   (float[36], histogram, histogram)
);

namespace pcl
{
  /** \brief Supervoxel container class - stores a cluster extracted using
   * supervoxel clustering
   */
  template <typename PointT>
  class SequentialSV : public Supervoxel<PointT>
  {
    public:
      SequentialSV(bool is_new):is_new_(is_new)
      {

      }

      bool
      isNew() const
      { return is_new_; }

      using Supervoxel<PointT>::Supervoxel;

      typedef boost::shared_ptr<SequentialSV> Ptr;
      typedef boost::shared_ptr<const SequentialSV> ConstPtr;

      /** \brief The normal calculated for the voxels contained in the
       *  supervoxel */
      using Supervoxel<PointT>::normal_;
      /** \brief The centroid of the supervoxel - average voxel */
      using Supervoxel<PointT>::centroid_;
      /** \brief A Pointcloud of the voxels in the supervoxel */
      using Supervoxel<PointT>::voxels_;
      /** \brief A Pointcloud of the normals for the points in the supervoxel*/
      using Supervoxel<PointT>::normals_;
      /** \brief A Pointcloud of the voxels with xyzrgb+label data in it */
      pcl::PointCloud<pcl::PointXYZRGBL> labeled_voxels_;
    private:
      /** \brief Boolean telling whether or not the supervoxel is new */
      bool is_new_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  /** \brief Class for persistent supervoxel clustering, a clustering spanning
   * on multiples frames of a same scene
   *  \author Elias Hanna (h.elias@hotmail.fr) and Jeremie Papon
   * (jpapon@gmail.com)
   *  \ingroup segmentation
   */
  template <typename PointT>
  class PCL_EXPORTS SequentialSVClustering : public pcl::PCLBase<PointT>
  {
      class SequentialSupervoxelHelper;
      friend class SequentialSupervoxelHelper;
      friend class RansacSupervoxelTracker;
    public:
      // Attributes used for visualization
      std::unordered_map<uint32_t, std::pair<Eigen::Vector4f, Eigen::Vector4f>>
      lines_;
      std::vector<int> previous_keypoints_indices_;
      std::vector<int> current_keypoints_indices_;
      std::vector<pcl::PointXYZRGBA> centroid_of_dynamic_svs_;
      uint64_t frame_number_;
      std::vector<uint32_t>
      getLabelColors () const;

      /** \brief VoxelData is a structure used for storing data within a
       * pcl::octree::OctreePointCloudAdjacencyContainer
       *  \note It stores xyz, rgb, normal, distance, an index, and an owner.
       */
      class SequentialVoxelData : public SupervoxelClustering<PointT>::VoxelData
      {
        public:
          SequentialVoxelData ():
            new_leaf_ (true),
            has_changed_ (false),
            frame_occluded_ (0),
            label_ (-1)
          {
            idx_ = -1;
            // Initialize previous state of the voxel
            previous_xyz_ = xyz_;
            previous_rgb_ = rgb_;
            previous_normal_ = normal_;
          }

          bool
          isNew () const { return new_leaf_; }

          void
          setNew (bool new_arg) { new_leaf_ = new_arg; }

          bool
          isChanged () const { return has_changed_; }

          void
          setChanged (bool new_val) { has_changed_ = new_val; }

          void
          prepareForNewFrame ()
          {
            new_leaf_ = false;
            has_changed_ = false;
            // Update the previous state of the voxel
            previous_xyz_ = xyz_;
            previous_rgb_ = rgb_;
            previous_normal_ = normal_;
            xyz_ = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            rgb_ = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            normal_ = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
          }

          void
          revertToLastPoint ()
          {
            xyz_ = previous_xyz_;
            rgb_ = previous_rgb_;
            normal_ = previous_normal_;
          }

          void
          initLastPoint ()
          {
            previous_xyz_ = xyz_;
            previous_rgb_ = rgb_;
            previous_normal_ = normal_;
          }

          // Use the methods from VoxelData
          /* REMOVED TO SUPPORT MORE POINT TYPES */
          //          using SupervoxelClustering<PointT>::VoxelData::getPoint;
          //          using SupervoxelClustering<PointT>::VoxelData::getNormal;

          /** \brief Gets the data of in the form of a point
           *  \param[out] point_arg Contain the point value of the voxeldata
           */
          void
          getPoint (PointT &point_arg) const;

          /** \brief Gets the data of in the form of a normal
           *  \param[out] normal_arg Contain the normal value of the voxeldata
           */
          void
          getNormal (Normal &normal_arg) const;

          // Use the attributes from VoxelData
          using SupervoxelClustering<PointT>::VoxelData::idx_;
          using SupervoxelClustering<PointT>::VoxelData::xyz_;
          using SupervoxelClustering<PointT>::VoxelData::rgb_;
          using SupervoxelClustering<PointT>::VoxelData::normal_;
          using SupervoxelClustering<PointT>::VoxelData::curvature_;
          using SupervoxelClustering<PointT>::VoxelData::distance_;

          // Used by the difference function
          Eigen::Vector3f previous_xyz_;
          Eigen::Vector3f previous_rgb_;
          Eigen::Vector4f previous_normal_;

          // New attributes
          bool has_changed_, new_leaf_;
          int frame_occluded_;
          int label_;
          SequentialSupervoxelHelper* owner_;

        public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      };

      typedef
      pcl::octree::OctreePointCloudSequentialContainer
      <PointT, SequentialVoxelData> LeafContainerT;
      typedef std::vector <LeafContainerT*> LeafVectorT;
      typedef
      std::map<uint32_t,typename Supervoxel<PointT>::Ptr> SupervoxelMapT;
      typedef
      std::map<uint32_t,typename SequentialSV<PointT>::Ptr> SequentialSVMapT;

      typedef typename pcl::PointCloud<PointT> PointCloudT;
      typedef typename pcl::PointCloud<Normal> NormalCloud;
      typedef
      typename pcl::octree::OctreePointCloudSequential<PointT, LeafContainerT>
      OctreeSequentialT;
      typedef
      typename pcl::octree::OctreePointCloudSearch <PointT> OctreeSearchT;
      typedef typename pcl::search::KdTree<PointT> KdTreeT;
      typedef boost::shared_ptr<std::vector<int> > IndicesPtr;

      // Keypoints and descriptors types
      typedef pcl::PointCloud<pcl::PointWithScale> PointCloudScale;
      typedef pcl::PointCloud<pcl::IntensityGradient> PointCloudIG;
      typedef pcl::PointCloud<pcl::PointXYZI> PointCloudI;

      // Feature space search types
//      typedef pcl::FPFHSignature33 FeatureT;
      typedef pcl::FPFHCIELabSignature36 FeatureT;
      typedef flann::L1<float> DistanceT; // Manhattan distance
      //typedef flann::L2<float> DistanceT;
      //typedef flann::KL_Divergence<float> DistanceT;
      //typedef flann::MinkowskiDistance<float> DistanceT;
      //typedef flann::HistIntersectionDistance<float> DistanceT;
      typedef pcl::PointCloud<FeatureT> PointCloudFeatureT;
      typedef
      std::pair<pcl::IndicesPtr, PointCloudFeatureT::Ptr> KeypointFeatureT;
      typedef
      std::unordered_map<uint32_t, KeypointFeatureT> KeypointMapFeatureT;

      // Search and index types
      typedef search::FlannSearch<FeatureT, DistanceT> SearchT;
      typedef typename SearchT::FlannIndexCreatorPtr CreatorPtrT;
      typedef typename SearchT::KdTreeMultiIndexCreator IndexT;
      typedef typename SearchT::PointRepresentationPtr RepresentationPtrT;

      using pcl::PCLBase <PointT>::initCompute;
      using pcl::PCLBase <PointT>::deinitCompute;
      using pcl::PCLBase <PointT>::input_;

      typedef
      boost::adjacency_list
      <boost::setS, boost::setS, boost::undirectedS, uint32_t, float>
      VoxelAdjacencyList;
      typedef VoxelAdjacencyList::vertex_descriptor VoxelID;
      typedef VoxelAdjacencyList::edge_descriptor EdgeID;

    public:
      /** \brief Constructor that sets default values for member variables.
       *  \param[in] voxel_resolution The resolution (in meters) of voxels used
       *  \param[in] seed_resolution The average size (in meters) of resulting
       * supervoxels
       *  \param[in] use_single_camera_transform Set to true if point density
       * in cloud falls off with distance from origin (such as with a cloud
       * coming from one stationary camera), set false if input cloud is from
       * multiple captures from multiple locations.
       */
      SequentialSVClustering (float voxel_resolution, float seed_resolution,
                              bool use_single_camera_transform = true,
                              bool prune_close_seeds=true);

      /** \brief This destructor destroys the cloud, normals and search
       * method used for finding neighbors. In other words it frees memory.
       */
      virtual
      ~SequentialSVClustering ();

      /** \brief Set the resolution of the octree voxels */
      void
      setVoxelResolution (float resolution);

      /** \brief Get the resolution of the octree voxels */
      float
      getVoxelResolution () const;

      /** \brief Set the resolution of the octree seed voxels */
      void
      setSeedResolution (float seed_resolution);

      /** \brief Get the resolution of the octree seed voxels */
      float
      getSeedResolution () const;

      /** \brief Set the importance of color for supervoxels */
      void
      setColorImportance (float val);

      /** \brief Set the importance of spatial distance for supervoxels */
      void
      setSpatialImportance (float val);

      /** \brief Set the importance of scalar normal product for supervoxels */
      void
      setNormalImportance (float val);

      /** \brief Set whether or not to use the single camera transform
       *  \note By default it will be used for organized clouds, but not for
       * unorganized - this parameter will override that behavior
       *  The single camera transform scales bin size so that it increases
       * exponentially with depth (z dimension).
       *  This is done to account for the decreasing point density found with
       * depth when using an RGB-D camera.
       *  Without the transform, beyond a certain depth adjacency of voxels
       * breaks down unless the voxel size is set to a large value.
       *  Using the transform allows preserving detail up close, while allowing
       *  adjacency at distance.
       *  The specific transform used here is:
       *  x /= z; y /= z; z = ln(z);
       *  This transform is applied when calculating the octree bins in
       * OctreePointCloudAdjacency
       */
      void
      setUseSingleCameraTransform (bool val);

      /** \brief Set to ignore input normals and calculate normals internally
       *  \note Default is False - ie, SupervoxelClustering will use normals
       * provided in PointT if there are any
       *  \note You should only need to set this if eg PointT=PointXYZRGBNormal
       *  but you don't want to use the normals it contains
       */
      void
      setIgnoreInputNormals (bool val);

      /** \brief Returns the current maximum (highest) label */
      int
      getMaxLabel () const;

      /** \brief This method launches the segmentation algorithm and returns
       * the supervoxels that were
       * obtained during the segmentation.
       * \param[out] supervoxel_clusters A map of labels to pointers to
       * supervoxel structures
       */
      virtual void
      extract (SequentialSVMapT &supervoxel_clusters);

      /** \brief This method sets the cloud to be supervoxelized
       * \param[in] cloud The cloud to be supervoxelize
       */
      virtual void
      setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr& cloud);

      /** \brief This method sets the normals to be used for supervoxels
       *  (should be same size as input cloud)
      * \param[in] normal_cloud The input normals
      */
      virtual void
      setNormalCloud (typename NormalCloud::ConstPtr normal_cloud);

      pcl::PointCloud<pcl::PointXYZL>::Ptr
      getLabeledCloud () const;

      pcl::PointCloud<pcl::PointXYZL>::Ptr
      getLabeledVoxelCloud () const;

      pcl::PointCloud<pcl::PointXYZRGBL>::Ptr
      getLabeledRGBVoxelCloud () const;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
      getColoredVoxelCloud () const;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
      getColoredCloud () const;

      /** \brief Returns a deep copy of the voxel centroid cloud */
      typename pcl::PointCloud<PointT>::Ptr
      getVoxelCentroidCloud () const;

      /** \brief Returns a deep copy of the previous voxel centroid cloud */
      typename pcl::PointCloud<PointT>::Ptr
      getPrevVoxelCentroidCloud () const;

      /** \brief Returns a deep copy of the voxel centroid cloud */
      typename pcl::PointCloud<PointT>::Ptr
      getUnlabeledVoxelCentroidCloud () const;

      /** \brief Returns a deep copy of the voxel normal cloud */
      pcl::PointCloud<pcl::Normal>::Ptr
      getVoxelNormalCloud () const;

      /** \brief Returns a deep copy of the unlabeled voxel normal cloud */
      pcl::PointCloud<pcl::Normal>::Ptr
      getUnlabeledVoxelNormalCloud () const;

      /** \brief Gets the adjacency list (Boost Graph library) which gives
       * connections between supervoxels
       *  \param[out] adjacency_list_arg BGL graph where supervoxel labels are
       * vertices, edges are touching relationships
       */
      void
      getSupervoxelAdjacency
      (std::multimap<uint32_t, uint32_t>& label_adjacency) const;

      /**
       * \brief getMovingParts
       * \return
       */
      std::vector<uint32_t>
      getMovingParts ()
      { return moving_parts_; }

      /**
       * \brief getToResetParts
       * \return
       */
      std::vector<uint32_t>
      getToResetParts ()
      { return to_reset_parts_; }

      /**
       * \brief getMovingParts
       * \param vec
       */
      void
      getMovingParts (std::vector<uint32_t>& vec)
      { vec = moving_parts_; }

      /**
       * \brief getToResetParts
       * \param vec
       */
      void
      getToResetParts (std::vector<uint32_t>& vec)
      { vec = to_reset_parts_; }

    private:
      /** \brief This method initializes the label_colors_ vector
       * (assigns random colors to labels)
       * \note Checks to see if it is already big enough - if so, does not
       * reinitialize it
       */
      void
      initializeLabelColors ();

      /** \brief This method computes the FPFH descriptors of the given
       * keypoints
       * \param[out] A pointcloud of the descriptors
       */
      std::pair< pcl::IndicesPtr, pcl::PointCloud<pcl::FPFHSignature33>::Ptr >
      computeFPFHDescriptors (const PointCloudScale sift_result,
                              const typename PointCloudT::Ptr cloud,
                              const NormalCloud::Ptr normals) const;


      std::pair
      <pcl::IndicesPtr,pcl::PointCloud<pcl::FPFHCIELabSignature36>::Ptr>
      computeFPFHCIELabDescriptors
      (const PointCloudScale sift_result, const typename PointCloudT::Ptr cloud,
       const NormalCloud::Ptr normals) const;

      /** \brief This method filters out the keypoints where the descriptor
       * don't hold enough information
       *  \note Should the descriptor always hold good information ? Don't
       * know if it is a metaparameter
       * problem related to the use of the RIFTEstimation class or if it is
       * normal. */
      pcl::PointIndicesPtr
      filterKeypoints
      (const std::pair<pcl::IndicesPtr, PointCloudFeatureT::Ptr>
       to_filter_keypoints,
       std::pair<pcl::IndicesPtr, PointCloudFeatureT::Ptr>& filtered_keypoints)
      const;

      /** \brief This method compute key points matches between an input vector
       * of potential inliers and points in an input pointcloud
       * \note This method set priority over order of preference rather than
       * distance between point and potential match
       * \param[out] The vector of point indices that match the input indices,
       * out[i]=-1 if no matching point was found */
      std::vector<int>
      computeKeypointsMatches(const std::vector<int> to_match_indices,
                              const PointCloudFeatureT to_match_feature_cloud,
                              const std::pair <pcl::IndicesPtr,
                              PointCloudFeatureT::Ptr > indices_point_pair);

      /** \brief Init the computation, update the sequential octree, perform
       * global check to see wether supervoxel have changed more than their
       * half and finally compute the voxel data to be used to determine the
       * supervoxel seeds
       */
      void
      buildVoxelCloud ();

      /** \brief Update the sequential octree, perform global check to see
       * wether supervoxel have changed more than their half and finally
       * compute the voxel data to be used to determine the supervoxel seeds
       */
      bool
      prepareForSegmentation ();

      /** \brief This method unlabels changed voxels between two frames and
       * also unlabel more than half changing supervoxels
       */
      void
      globalCheck ();

      /** \brief This method computes the label of the occluded/disappeared
       * supervoxels and returns a vector of those labels
       * \param[out] Vector of labels (uint32_t) */
      std::vector<uint32_t>
      getLabelsOfDynamicSV (SequentialSVMapT &supervoxel_clusters);

      /** \brief This methode removes the keypoints from current_keypoints
       * given by the indices in to remove_indices */
      void
      removeInliers (KeypointFeatureT& current_keypoints,
                     const std::vector<int>& to_remove_indices);

      /**
       * \brief testForOcclusionCriterium
       * \param to_match_indices
       * \param matches_indices
       * \return
       */
      bool
      testForOcclusionCriterium (const std::vector<int>& to_match_indices,
                                 const std::vector<int>& matches_indices)const;

      /** \brief This method uses a RANSAC based algorithm to find matches to
       * disappeared/occluded supervoxels from previous frame that woud appear
       *  in the current frame
       * \param[out] matches found in the form of an STL unordered map with
       * label as key and pcl::recognition::ObjRecRANSAC::Output as value
       */
//      std::unordered_map<uint32_t, Eigen::Matrix<float, 4, 4>>
      std::unordered_map<uint32_t, std::vector<float>>
      getMatchesRANSAC (SequentialSVMapT &supervoxel_clusters);

      /** \brief This method computes the keypoints and the descriptors of the
       * input point cloud and stores them
       * \note This overload compute keypoints and descriptors from the
       * previous supervoxel clusters and stores it as a map linking the
       * supervoxel label to a pair of indices (the keypoints in previous voxel
       *  cloud) and the corresponding descriptor cloud (FeatureT points)
       * \param[out] returns true if keypoints have been found in at least one
       * of the previous voxel clouds corresponding to a supervoxel */
      bool
      computeUniformKeypointsAndFPFHDescriptors
      (SequentialSVMapT &supervoxel_clusters,
       KeypointMapFeatureT &previous_keypoints,
       int min_nb_of_keypoints);

      /** \brief This method computes the keypoints and the descriptors of the
       * input point cloud and stores them
       * \note This overload compute keypoints and descriptors from the current
       *  unlabeled voxel cloud and stores it as a pair of indices
       * (the keypoints in previous voxel cloud) and the corresponding
       * descriptor cloud (FeatureT points)
       * \param[out] returns true if keypoints have been found in the current
       * voxel cloud */
      bool
      computeUniformKeypointsAndFPFHDescriptors
      (KeypointFeatureT &current_keypoints,
       int min_nb_of_keypoints);

      /** \brief This method computes the keypoints and the descriptors of the
       * input point cloud and stores them
       * \note This overload compute keypoints and descriptors from the
       * previous supervoxel clusters and stores it as a map linking the
       * supervoxel label to a pair of indices (the keypoints in previous voxel
       *  cloud) and the corresponding descriptor cloud (FeatureT points)
       * \param[out] returns true if keypoints have been found in at least one
       * of the previous voxel clouds corresponding to a supervoxel */
      bool
      computeUniformKeypointsAndFPFHCIELabDescriptors
      (SequentialSVMapT &supervoxel_clusters,
       KeypointMapFeatureT &previous_keypoints,
       int min_nb_of_keypoints);

      /** \brief This method computes the keypoints and the descriptors of the
       * input point cloud and stores them
       * \note This overload compute keypoints and descriptors from the current
       *  unlabeled voxel cloud and stores it as a pair of indices
       * (the keypoints in previous voxel cloud) and the corresponding
       * descriptor cloud (FeatureT points)
       * \param[out] returns true if keypoints have been found in the current
       * voxel cloud */
      bool
      computeUniformKeypointsAndFPFHCIELabDescriptors
      (KeypointFeatureT &current_keypoints,
       int min_nb_of_keypoints);

      /** \brief This method randomly draws nb_to_sample points from indices
       * (and removes them from indices and cloud) and stores them accordingly
       * in potential_inliers and potential_inliers_feature_cloud. */
      void
      samplePotentialInliers
      (std::vector<int> &indices,
       PointCloudFeatureT::Ptr &cloud,
       std::vector<int> &potential_inliers,
       PointCloudFeatureT::Ptr &potential_inliers_feature_cloud,
       int nb_to_sample);

      /** \brief Compute the rigid transformation between the corresponding
       * point indices designated by the two input vectors. Uses
       * TransformationSVD to do so
       * \note could add an overloaded method to accept pcl::Correspondances */
      Eigen::Matrix<float, 4, 4>
      computeRigidTransformation(std::vector<int> prev_indices,
                                 std::vector<int> curr_indices);

      /** \brief This method search in spatial_neighbours which ones are
       * keypoints and adds them as inliers if their distance in feature space
       * is below a threshold */
      void
      findInliers(const PointCloudFeatureT::Ptr &search_cloud,
                  const std::vector<int> &spatial_neighbors,
                  const std::vector<int> &all_indices,
                  const PointCloudFeatureT::Ptr &all_cloud,
                  std::vector<int> &potential_inliers,
                  PointCloudFeatureT::Ptr &potential_inliers_cloud,
                  float threshold, float* err);

      /** \brief Compute the voxel data (index of each voxel in the octree and
       * normal of each voxel) */
      void
      computeVoxelData ();

      /**
       * \brief computeUnlabeledVoxelCentroidNormalCloud
       */
      void
      computeUnlabeledVoxelCentroidNormalCloud ();


      /** \brief Compute the voxel data (index of each voxel in the octree and
       * normal of each voxel) */
      void
      updatePrevClouds ();

      /** \brief Update the unlabeled voxel and normal cloud by removing the
       * indices given by indices
        * and using pcl::StatisticalOutlierRemoval to remove noise from
        * unlabeled voxel cloud */
      void
      updateUnlabeledCloud ();

      /** \brief Update the unlabeled normal cloud by removing the indices
       * given by indices */
      void
      updateUnlabeledNormalCloud (const IndicesConstPtr indices);

      /** \brief This method compute the normal of each leaf belonging to
       * the sequential octree
       */
      void
      parallelComputeNormals ();

      /**
       * \brief computeUnlabeledVoxelCentroidCloud
       */
      void
      computeUnlabeledVoxelCentroidCloud ();

      /** \brief Distance function used for comparing voxelDatas */
      float
      sequentialVoxelDataDistance (const SequentialVoxelData &v1,
                                   const SequentialVoxelData &v2) const;

      /** \brief Transform function used to normalize voxel density versus
       * distance from camera */
      void
      transformFunction (PointT &p);

      /** \brief This roughly founds the same seeding points as those from
       * the previous frame
       *  \param[out] existing_seed_indices The selected leaf indices
       */
      void
      getPreviousSeedingPoints (SequentialSVMapT &supervoxel_clusters,
                                std::vector<int> &existing_seed_indices);

      /** \brief This method finds seeding points, then prune the seeds that
       * are too close to existing ones and stores seeds that are going to be
       * used to supervoxelize the scene in seed_indices
       */
      void
      pruneSeeds (std::vector<int> &existing_seed_indices,
                  std::vector<int> &seed_indices);

      /** \brief This performs the superpixel evolution */
      void
      expandSupervoxels ( int depth );

      /** \brief Constructs the map of supervoxel clusters from the internal
       * supervoxel helpers */
      void
      makeSupervoxels (SequentialSVMapT &supervoxel_clusters);

      void
      addHelpersFromUnlabeledSeedIndices(std::vector<int> &seed_indices);

      std::vector<int>
      getAvailableLabels ();

      /** \brief Stores the resolution used in the octree */
      float resolution_;

      /** \brief Stores the resolution used to seed the supervoxels */
      float seed_resolution_;

      /** \brief Contains a KDtree for the voxelized cloud */
      typename pcl::search::KdTree<PointT>::Ptr voxel_kdtree_;

      /** \brief Stores the colors used for the superpixel labels*/
      std::vector<uint32_t> label_colors_;

      /** \brief Octree Sequential structure with leaves at voxel resolution */
      typename OctreeSequentialT::Ptr sequential_octree_;

      /** \brief Octree Sequential structure with leaves at voxel resolution */
      typename OctreeSequentialT::Ptr prev_sequential_octree_;

      /** \brief Contains the Voxelized centroid cloud of the unlabeled
       * voxels */
      typename PointCloudT::Ptr unlabeled_voxel_centroid_cloud_;

      /** \brief Contains the Normals of the current unlabeled voxel centroid
       * cloud */
      typename NormalCloud::Ptr unlabeled_voxel_centroid_normal_cloud_;

      /** \brief Contains the Voxelized centroid Cloud */
      typename PointCloudT::Ptr voxel_centroid_cloud_;

      /** \brief Contains the Voxelized centroid Cloud of the previous frame*/
      typename PointCloudT::Ptr prev_voxel_centroid_cloud_;

      /** \brief Contains the Normals of the voxel centroid Cloud of the
       * previous frame*/
      typename NormalCloud::Ptr prev_voxel_centroid_normal_cloud_;

      /**
       * \brief prev_keypoints_location_
       */
      std::unordered_map<uint32_t, pcl::PointCloud<pcl::PointXYZ>::Ptr>
      prev_keypoints_location_;
      /**
       * \brief curr_keypoints_location_
       */
      pcl::PointCloud<pcl::PointXYZ>::Ptr
      curr_keypoints_location_;

      /** \brief Contains the Normals of the input Cloud */
      typename NormalCloud::ConstPtr input_normals_;

      /** \brief Importance of color in clustering */
      float color_importance_;

      /** \brief Importance of distance from seed center in clustering */
      float spatial_importance_;

      /** \brief Importance of similarity in normals for clustering */
      float normal_importance_;

      /** \brief Option to ignore normals in input Pointcloud. Defaults to
       * false */
      bool ignore_input_normals_;

      /** \brief Whether or not to use the transform compressing depth in Z
       *  This is only checked if it has been manually set by the user.
       *  The default behavior is to use the transform for organized, and not
       * for unorganized. */
      bool use_single_camera_transform_;

      /** \brief Whether to use default transform behavior or not */
      bool use_default_transform_behaviour_;

      bool prune_close_seeds_;

      pcl::StopWatch timer_;
      boost::mutex mutex_normals_;

      int nb_of_unlabeled_voxels_;

      std::vector<uint32_t> moving_parts_;

      std::vector<uint32_t> to_reset_parts_;
      /** \brief Internal storage class for supervoxels
       * \note Stores pointers to leaves of clustering internal octree,
       * \note so should not be used outside of clustering class
       */
      class SequentialSupervoxelHelper
      {
        public:

          /** \brief Comparator for LeafContainerT pointers - used for sorting
           * set of leaves
         * \note Compares by index in the overall leaf_vector. Order isn't
         * important, so long as it is fixed.
         */
          struct compareLeaves
          {
              bool operator() (LeafContainerT* const &left,
                               LeafContainerT* const &right) const
              {
                const SequentialVoxelData& leaf_data_left = left->getData ();
                const SequentialVoxelData& leaf_data_right = right->getData ();
                return leaf_data_left.idx_ < leaf_data_right.idx_;
              }
          };
          typedef
          std::set
          <LeafContainerT*, typename SequentialSupervoxelHelper::compareLeaves>
          LeafSetT;
          typedef typename LeafSetT::iterator iterator;
          typedef typename LeafSetT::const_iterator const_iterator;

          SequentialSupervoxelHelper (uint32_t label,
                                      SequentialSVClustering* parent_arg):
            label_ (label),
            parent_ (parent_arg),
            is_new_ (true)
          { }

          void
          addLeaf (LeafContainerT* leaf_arg);

          void
          removeLeaf (LeafContainerT* leaf_arg);

          void
          removeAllLeaves ();

          void
          expand ();

          void
          updateCentroid ();

          bool
          isNew () const
          { return is_new_; }

          bool
          setNew (bool is_new)
          { is_new_ = is_new; }

          void
          getVoxels (typename pcl::PointCloud<PointT>::Ptr &voxels) const;

          void
          getNormals (typename pcl::PointCloud<Normal>::Ptr &normals) const;

          typedef float (SequentialSVClustering::*DistFuncPtr)
          (const SequentialVoxelData &v1, const SequentialVoxelData &v2);

          uint32_t
          getLabel () const
          { return label_; }

          Eigen::Vector4f
          getNormal () const
          { return centroid_.normal_; }

          Eigen::Vector3f
          getRGB () const
          { return centroid_.rgb_; }

          Eigen::Vector3f
          getXYZ () const
          { return centroid_.xyz_;}

          void
          getXYZ (float &x, float &y, float &z) const
          { x=centroid_.xyz_[0]; y=centroid_.xyz_[1]; z=centroid_.xyz_[2]; }

          void
          getRGB (uint32_t &rgba) const
          {
            rgba = static_cast<uint32_t>( centroid_.rgb_[0])
                << 16 | static_cast<uint32_t>(centroid_.rgb_[1])
                << 8 | static_cast<uint32_t>(centroid_.rgb_[2]);
          }

          void
          getNormal (pcl::Normal &normal_arg) const
          {
            normal_arg.normal_x = centroid_.normal_[0];
            normal_arg.normal_y = centroid_.normal_[1];
            normal_arg.normal_z = centroid_.normal_[2];
            normal_arg.curvature = centroid_.curvature_;
          }

          void
          getNeighborLabels (std::set<uint32_t> &neighbor_labels) const;

          SequentialVoxelData
          getCentroid () const
          {
            return centroid_;
          }

          size_t
          size () const { return leaves_.size (); }
        private:
          //Stores leaves
          LeafSetT leaves_;
          uint32_t label_;
          SequentialVoxelData centroid_;
          SequentialSVClustering* parent_;
          bool is_new_;
        public:
          //Type VoxelData may have fixed-size Eigen objects inside
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      };

      //Make boost::ptr_list can access the private class SupervoxelHelper
      friend void boost::checked_delete<>
      (const typename
       pcl::SequentialSVClustering<PointT>::SequentialSupervoxelHelper *);

      typedef boost::ptr_list<SequentialSupervoxelHelper> HelperListT;
      HelperListT supervoxel_helpers_;

      class RANSACRegistration {
          const KeypointFeatureT& current_keypoints_;
          const KeypointMapFeatureT::value_type& pair_;
          const int min_number_of_inliers_;
          const SequentialSVMapT& supervoxel_clusters_;
          const typename PointCloudT::Ptr prev_voxel_centroid_cloud_;
          const typename PointCloudT::Ptr unlabeled_voxel_centroid_cloud_;
          const typename PointCloudT::Ptr voxel_centroid_cloud_;
          const float threshold_;
          const float seed_resolution_;
        public:
          std::vector<int> best_inliers_set_;
          float best_score_;
          Eigen::Matrix<float, 4, 4> best_fit_;

          void operator()( const tbb::blocked_range<size_t>& r );

          RANSACRegistration( RANSACRegistration& x, tbb::split ):
            current_keypoints_ (x.current_keypoints_),
            pair_ (x.pair_),
            min_number_of_inliers_ (x.min_number_of_inliers_),
            supervoxel_clusters_ (x.supervoxel_clusters_),
            prev_voxel_centroid_cloud_ (x.prev_voxel_centroid_cloud_),
            unlabeled_voxel_centroid_cloud_(x.unlabeled_voxel_centroid_cloud_),
            voxel_centroid_cloud_ (x.voxel_centroid_cloud_),
            threshold_ (x.threshold_),
            seed_resolution_ (x.seed_resolution_),
            best_score_ (0)
          {}

          void join( const RANSACRegistration& y )
          {
            if ( y.best_score_ > best_score_)
            {
              best_score_ = y.best_score_;
              best_fit_ = y.best_fit_;
              best_inliers_set_ = y.best_inliers_set_;
            }
          }

          RANSACRegistration (const KeypointFeatureT& current_keypoints,
                              const KeypointMapFeatureT::value_type& pair,
                              const int min_number_of_inliers,
                              const SequentialSVMapT& supervoxel_clusters,
                              const typename PointCloudT::Ptr
                              prev_voxel_centroid_cloud,
                              const typename PointCloudT::Ptr
                              unlabeled_voxel_centroid_cloud,
                              const typename PointCloudT::Ptr
                              voxel_centroid_cloud,
                              const float threshold,
                              const float seed_resolution):
            current_keypoints_ (current_keypoints),
            pair_ (pair),
            min_number_of_inliers_ (min_number_of_inliers),
            supervoxel_clusters_ (supervoxel_clusters),
            prev_voxel_centroid_cloud_ (prev_voxel_centroid_cloud),
            unlabeled_voxel_centroid_cloud_ (unlabeled_voxel_centroid_cloud),
            voxel_centroid_cloud_ (voxel_centroid_cloud),
            threshold_ (threshold),
            seed_resolution_ (seed_resolution),
            best_score_ (0)
          {}

        private:
          void
          samplePotentialInliers(std::vector<int> &indices,
                                 PointCloudFeatureT::Ptr &cloud,
                                 std::vector<int> &potential_inliers,
                                 PointCloudFeatureT::Ptr&
                                 potential_inliers_feature_cloud,
                                 int nb_to_sample);

          Eigen::Matrix<float, 4, 4>
          computeRigidTransformation(std::vector<int> prev_indices,
                                     std::vector<int> curr_indices);

          void
          findInliers(const PointCloudFeatureT::Ptr &search_cloud,
                      const std::vector<int> &spatial_neighbors,
                      const std::vector<int> &all_indices,
                      const PointCloudFeatureT::Ptr &all_cloud,
                      std::vector<int> &potential_inliers,
                      PointCloudFeatureT::Ptr &potential_inliers_cloud,
                      float threshold, float* err);

          std::vector<int>
          computeKeypointsMatches(const std::vector<int> to_match_indices,
                                  const PointCloudFeatureT
                                  to_match_feature_cloud,
                                  const std::pair
                                  <pcl::IndicesPtr, PointCloudFeatureT::Ptr>
                                  indices_point_pair);
      };

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

}

#ifdef PCL_NO_PRECOMPILE
#include "impl/sequential_supervoxel_clustering.hpp"
#endif

#endif //PCL_SEGMENTATION_SEQUENTIAL_SUPERVOXEL_CLUSTERING_H_
