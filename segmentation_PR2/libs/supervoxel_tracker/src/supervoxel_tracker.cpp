#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>
/* Includes for instantiations */
#include <pcl/tracking/impl/hsv_color_coherence.hpp>

#include <pcl/tracking/impl/particle_filter.hpp>
#include <pcl/tracking/impl/particle_filter_omp.hpp>

#include <pcl/tracking/impl/kld_adaptive_particle_filter.hpp>
#include <pcl/tracking/impl/kld_adaptive_particle_filter_omp.hpp>

#include "../impl/supervoxel_tracker.hpp"
#include "../impl/label_coherence.hpp"
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

// THOSE DECLARATIONS SHOULD BE ADDED TO COHERENCE.CPP

/* Explicit instantiations for LabelCoherence class */
template class pcl::tracking::LabelCoherence<pcl::PointXYZRGBL>;
/* Explicit instantiations for HSVColorCoherence class */
template class pcl::tracking::HSVColorCoherence<pcl::PointXYZRGBL>;

/* Explicit instantiations for ParticleFilterTracker class */
template class pcl::tracking::ParticleFilterTracker<pcl::PointXYZRGBL, pcl::tracking::ParticleXYZRPY>;
/* Explicit instantiations for ParticleFilterOMPTracker class */
template class pcl::tracking::ParticleFilterOMPTracker<pcl::PointXYZRGBL, pcl::tracking::ParticleXYZRPY>;
/* Explicit instantiations for KLDAdaptiveParticleFilterTracker class */
template class pcl::tracking::KLDAdaptiveParticleFilterTracker<pcl::PointXYZRGBL, pcl::tracking::ParticleXYZRPY>;
/* Explicit instantiations for KLDAdaptiveParticleFilterOMPTracker class */
template class pcl::tracking::KLDAdaptiveParticleFilterOMPTracker<pcl::PointXYZRGBL, pcl::tracking::ParticleXYZRPY>;

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

namespace pcl
{
  // Explicit overload for XYZRGB type
  template <> void
  pcl::SupervoxelTracker<pcl::PointXYZRGB, pcl::tracking::ParticleXYZRPY>::setupCoherences (typename pcl::tracking::PointCloudCoherence<pcl::PointXYZRGB>::Ptr coherence)
  {
    typename pcl::tracking::DistanceCoherence<pcl::PointXYZRGB>::Ptr distance_coherence (new pcl::tracking::DistanceCoherence<pcl::PointXYZRGB>);
    coherence->addPointCoherence (distance_coherence);

    typename pcl::tracking::HSVColorCoherence<pcl::PointXYZRGB>::Ptr hsv_color_coherence (new pcl::tracking::HSVColorCoherence<pcl::PointXYZRGB>);
    coherence->addPointCoherence (hsv_color_coherence);
  }

  // Explicit overload for XYZRGBA type
  template <> void
  pcl::SupervoxelTracker<pcl::PointXYZRGBA, pcl::tracking::ParticleXYZRPY>::setupCoherences (typename pcl::tracking::PointCloudCoherence<pcl::PointXYZRGBA>::Ptr coherence)
  {
    typename pcl::tracking::DistanceCoherence<pcl::PointXYZRGBA>::Ptr distance_coherence (new pcl::tracking::DistanceCoherence<pcl::PointXYZRGBA>);
    coherence->addPointCoherence (distance_coherence);

    typename pcl::tracking::HSVColorCoherence<pcl::PointXYZRGBA>::Ptr hsv_color_coherence (new pcl::tracking::HSVColorCoherence<pcl::PointXYZRGBA>);
    coherence->addPointCoherence (hsv_color_coherence);
  }

  // Explicit overload for XYZRGBL type
  template <> void
  pcl::SupervoxelTracker<pcl::PointXYZRGBL, pcl::tracking::ParticleXYZRPY>::setupCoherences (typename pcl::tracking::PointCloudCoherence<pcl::PointXYZRGBL>::Ptr coherence)
  {
    typename pcl::tracking::DistanceCoherence<pcl::PointXYZRGBL>::Ptr distance_coherence (new pcl::tracking::DistanceCoherence<pcl::PointXYZRGBL>);
    coherence->addPointCoherence (distance_coherence);

    typename pcl::tracking::HSVColorCoherence<pcl::PointXYZRGBL>::Ptr hsv_color_coherence (new pcl::tracking::HSVColorCoherence<pcl::PointXYZRGBL>);
    coherence->addPointCoherence (hsv_color_coherence);

    typename pcl::tracking::LabelCoherence<pcl::PointXYZRGBL>::Ptr label_coherence (new pcl::tracking::LabelCoherence<pcl::PointXYZRGBL>);
    coherence->addPointCoherence (label_coherence);
  }
}
//    if(pcl::traits::has_normal<PointT>::value)
//    {
//      typename pcl::tracking::NormalCoherence<PointT>::Ptr normal_coherence (new pcl::tracking::NormalCoherence<PointT>);
//      coherence->addPointCoherence (normal_coherence);
//    }



/* Explicit instantiations for SupervoxelTracker class */
//template class pcl::SupervoxelTracker<pcl::PointXYZ, pcl::tracking::ParticleXYZRPY>;
template class pcl::SupervoxelTracker<pcl::PointXYZRGB, pcl::tracking::ParticleXYZRPY>;
template class pcl::SupervoxelTracker<pcl::PointXYZRGBL, pcl::tracking::ParticleXYZRPY>;
template class pcl::SupervoxelTracker<pcl::PointXYZRGBA, pcl::tracking::ParticleXYZRPY>;
//PCL_INSTANTIATE_PRODUCT(SupervoxelTracker, ((pcl::PointXYZ)(pcl::PointXYZRGB)(pcl::PointXYZRGBA))(pcl::tracking::ParticleXYZRPY));
