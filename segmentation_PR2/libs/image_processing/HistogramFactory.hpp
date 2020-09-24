#ifndef HISTOGRAM_FACTORY_HPP
#define HISTOGRAM_FACTORY_HPP

#include <map>
#include <vector>
//#include "image_processing/pcl_types.h"
#include "pcl_types.h"
#include <memory>
#include <Eigen/Core>
#include "tools.hpp"

#include "SupervoxelSet.h"

namespace image_processing {

class HistogramFactory {

public:

    typedef std::vector<Eigen::VectorXd> _histogram_t;

    HistogramFactory(int bins, int dim, Eigen::MatrixXd bounds) :
        _bins(bins), _dim(dim), _bounds(bounds){
    }
    HistogramFactory(const HistogramFactory& HF)
        : _bins(HF._bins), _dim(HF._dim), _bounds(HF._bounds),
          _histogram(HF._histogram){}



    /**
     * @brief compute the <type> histogram related to sv.
     * For the color domain the histogram is computed on HSV encoding
     * @param sv
     * @param type {"color","normal"}
     */
    void compute(const pcl::Supervoxel<PointT>::ConstPtr& sv, std::string type = "color");
    /**
     * @brief compute the RGB color histograms of a open cv image
     * @param image
     */
    //void compute(const cv::Mat &image);

    /**
     * @brief compute the histograms of the set (data) of vectors according to the dim and the number of bins.
     * @param set of vectors
     */
    void compute(const std::vector<Eigen::VectorXd>& data);

    /**
     * @brief compute_multi_dim
     * @param data
     */
    void compute_multi_dim(const std::vector<Eigen::VectorXd>& data);

    /**
     * @brief chi_squared_distance
     * @param hist1
     * @param hist2
     * @return vector of distances
     */
    static double chi_squared_distance(const Eigen::VectorXd& hist1, const Eigen::VectorXd &hist2);


    //GETTERS & SETTERS

    /**
     * @brief return the histogram
     * @return histogram
     */
     _histogram_t get_histogram(){
        return _histogram;
    }


private:
    _histogram_t _histogram;

    int _bins;
    int _dim;
    Eigen::MatrixXd _bounds;

};

}

#endif //HISTOGRAM_FACTORY_HPP
