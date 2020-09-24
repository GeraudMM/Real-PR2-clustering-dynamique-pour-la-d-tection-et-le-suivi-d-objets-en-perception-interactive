#ifndef GETTER_HPP
#define GETTER_HPP

#include "getter.h"

template <typename PointType>
Getter<PointType>::Getter ():
  started_ (false)
{
}

template <typename PointType>
Getter<PointType>::Getter (boost::shared_ptr<pcl::Grabber> grabber):
  grabber_(grabber),
  started_(true)
{
  boost::function<void (const ConstPtr&)> callback = boost::bind(&Getter::cloud_callback, this, _1);
  connection_ = grabber_->registerCallback(callback);
  grabber_->start();
}

template <typename PointType>
Getter<PointType>::~Getter ()
{
  if (started_)
  {
    grabber_->stop();

    if(connection_.connected())
    {
      connection_.disconnect();
    }
  }
}

template <typename PointType> pcl::PointCloud<PointType>
Getter<PointType>::setGrabber (boost::shared_ptr<pcl::Grabber> grabber)
{
  grabber_ = grabber;
  boost::function<void (const ConstPtr&)> callback = boost::bind(&Getter::cloud_callback, this, _1);
  connection_ = grabber_->registerCallback(callback);
  grabber_->start();
  started_ = true;
}

template <typename PointType> pcl::PointCloud<PointType>
Getter<PointType>::getCloud ()
{
  return buffer_;
}

template <typename PointType> void
Getter<PointType>::cloud_callback (const ConstPtr& cloud)
{  
  buffer_ = *cloud;
}

#endif
