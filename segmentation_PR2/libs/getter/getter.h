#ifndef GETTER_H
#define GETTER_H

#include <pcl/io/openni_grabber.h>

template <typename PointType>
class Getter
{
  typedef pcl::PointCloud<PointType> PointCloud;
  typedef typename PointCloud::ConstPtr ConstPtr;

  private:
    boost::shared_ptr<pcl::Grabber> grabber_;
    boost::signals2::connection connection_;
    boost::mutex mutex_;
    pcl::PointCloud<PointType> buffer_;
    bool started_;

  public:
    Getter ();
    Getter (boost::shared_ptr<pcl::Grabber> grabber);
    ~Getter ();
    pcl::PointCloud<PointType> setGrabber(boost::shared_ptr<pcl::Grabber> grabber);
    pcl::PointCloud<PointType> getCloud ();

  private:
    void cloud_callback (const ConstPtr& cloud);
};

#endif
