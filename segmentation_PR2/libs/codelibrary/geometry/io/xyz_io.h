//
// Copyright 2016 Yangbin Lin. All Rights Reserved.
//
// Author: yblin@jmu.edu.cn (Yangbin Lin)
//
// This file is part of the Code Library.
//

#ifndef GEOMETRY_IO_XYZ_IO_H_
#define GEOMETRY_IO_XYZ_IO_H_

#include <fstream>
#include <sstream>

//#include "codelibrary/base/array.h"
#include "../../base/array.h"
//#include "codelibrary/base/log.h"
#include "../../base/log.h"
//#include "codelibrary/geometry/kernel/point_3d.h"
#include "../kernel/point_3d.h"
//#include "codelibrary/visualization/color/rgb32_color.h"
#include "../../visualization/color/rgb32_color.h"

namespace cl {
namespace geometry {
namespace io {

/**
 * Read points from XYZ file.
 */
template <typename T>
bool ReadXYZPoints(const char* filename, Array<Point3D<T> >* points) {
    assert(points);

    points->clear();

    std::ifstream in(filename);
    if (!in) {
        LOG(INFO) << "Cannot open XYZ file '" << filename << "' for reading.";
        return false;
    }

    int n_lines = 0;
    std::string line;
    T x, y, z;
    while (std::getline(in, line)) {
        std::istringstream is(line);

        if (!(is >> x) || !(is >> y) || !(is >> z)) {
            LOG(INFO) << "Invalid XYZ format at line: " << n_lines++;
            return false;
        }
        points->emplace_back(x, y, z);
    }

    return true;
}

/**
 * Read color points from XYZ file.
 */
template <typename T>
bool ReadXYZPoints(const char* filename, Array<Point3D<T> >* points,
                   Array<RGB32Color>* colors) {
    assert(points);
    assert(colors);

    points->clear();
    colors->clear();

    std::ifstream in(filename);
    if (!in) {
        LOG(INFO) << "Cannot open XYZ file '" << filename << "' for reading.";
        return false;
    }

    int n_lines = 0;
    std::string line;
    T x, y, z;
    int r, g, b;
    while (std::getline(in, line)) {
        std::istringstream is(line);

        if (!(is >> x) || !(is >> y) || !(is >> z) ||
            !(is >> r) || !(is >> g) || !(is >> b)) {
            LOG(INFO) << "Invalid XYZ format at line: " << n_lines++;
            in.close();
            return false;
        }
        points->emplace_back(x, y, z);
        colors->emplace_back(r, g, b);
    }

    in.close();

    return true;
}


/**
 * Read points color Label from XYZRGBL file.
 */
template <typename T>
bool ReadXYZPoints(const char* filename, Array<Point3D<T> >* points,
                   Array<Point3D<T> >* colors, Array<int>* labels) {
    assert(points);
    assert(colors);
    assert(labels);

    points->clear();
    colors->clear();
    labels->clear();

    std::ifstream in(filename);
    if (!in) {
        LOG(INFO) << "Cannot open XYZRGBL file '" << filename << "' for reading.";
        return false;
    }

    int n_lines = 0;
    std::string line;
    T x, y, z, r, g, b;
    int l;
    while (std::getline(in, line)) {
        std::istringstream is(line);

        if (!(is >> x) || !(is >> y) || !(is >> z) ||
            !(is >> r) || !(is >> g) || !(is >> b)|| !(is >> l)) {
            LOG(INFO) << "Invalid XYZRGBL format at line: " << n_lines++;
            in.close();
            return false;
        }
        points->emplace_back(x, y, z);
        colors->emplace_back(r, g, b);
        labels->emplace_back(l);
    }

    in.close();

    return true;
}

/**
 * Read oriented points from XYZ file.
 */
template <typename T>
bool ReadXYZPoints(const char* filename, Array<Point3D<T> >* points,
                   Array<RVector3D>* normals) {
    assert(points);
    assert(normals);

    points->clear();
    normals->clear();

    std::ifstream in(filename);
    if (!in) {
        LOG(INFO) << "Cannot open XYZ file '" << filename << "' for reading.";
        return false;
    }

    int n_lines = 0;
    std::string line;
    T x, y, z;
    double nx, ny, nz;
    while (std::getline(in, line)) {
        std::istringstream is(line);

        if (!(is >> x) || !(is >> y) || !(is >> z) ||
            !(is >> nx) || !(is >> ny) || !(is >> nz)) {
            LOG(INFO) << "Invalid XYZ format at line: " << n_lines++;
            in.close();
            return false;
        }
        points->emplace_back(x, y, z);
        normals->emplace_back(nx, ny, nz);
    }

    in.close();

    return true;
}

/**
 * Write points into XYZ file.
 */
template <typename T>
bool WriteXYZPoints(const std::string filename,
                    const Array<Point3D<T> >& points) {
    std::ofstream out(filename);
    if (!out) {
        LOG(INFO) << "Cannot open XYZ file '" << filename
                  << "' for writing.";
        return false;
    }

    out << std::setprecision(12);
    for (const Point3D<T>& p : points) {
        out << p.x << " " << p.y << " " << p.z << "\n";
    }

    out.close();

    return true;
}

/**
 * Write color points into XYZ file.
 */
template <typename T>
bool WriteXYZPoints(const std::string filename,
                    const Array<Point3D<T> >& points,
                    const Array<RGB32Color>& colors) {
    assert(points.size() == colors.size());

    std::ofstream out(filename);
    if (!out) {
        LOG(INFO) << "Cannot open XYZ file '" << filename
                  << "' for writing.";
        return false;
    }

    out << std::setprecision(12);
    for (int i = 0; i < points.size(); ++i) {
        const Point3D<T>& p = points[i];
        out << p.x << " " << p.y << " " << p.z << " ";
        const RGB32Color& c = colors[i];
        out << static_cast<int>(c.red()) << " " <<
               static_cast<int>(c.green()) << " " <<
               static_cast<int>(c.blue()) << "\n";
    }

    out.close();

    return true;
}

/**
 * Write points color label into XYZRGBL file.
 */
template <typename T>
bool WriteXYZPoints(const std::string filename,
                    const Array<Point3D<T> >& points,
                    const Array<Point3D<T> >& colors,
		    const Array<int>& label) {

    assert(points.size() == colors.size());

    std::ofstream out(filename);
    if (!out) {
        LOG(INFO) << "Cannot open XYZRGBL file '" << filename
                  << "' for writing.";
        return false;
    }

    out << std::setprecision(5);
    for (int i = 0; i < points.size(); ++i) {
        const Point3D<T>& p = points[i];
        out << p.x << " " << p.y << " " << p.z << " ";
	const Point3D<T>& c = colors[i];
        out << c.x << " " << c.y << " " << c.z << " ";
	out << label[i] <<"\n";
    }

    out.close();

    return true;
}

/**
 * Write points color label nSupervoxel into XYZRGBLS file.
 */
template <typename T>
bool WriteXYZPoints(const std::string filename,
                    const Array<Point3D<T> >& points,
                    const Array<RGB32Color>& colors,
		    const Array<int>& label, const Array<int>& numSv) {

    assert(points.size() == colors.size());

    std::ofstream out(filename);
    if (!out) {
        LOG(INFO) << "Cannot open XYZRGBLS file '" << filename
                  << "' for writing.";
        return false;
    }

    out << std::setprecision(5);
    for (int i = 0; i < points.size(); ++i) {
        const Point3D<T>& p = points[i];
        out << p.x << " " << p.y << " " << p.z << " ";
        const RGB32Color& c = colors[i];
        out << static_cast<int>(c.red()) << " " <<
               static_cast<int>(c.green()) << " " <<
               static_cast<int>(c.blue()) << " ";
	out << label[i] <<" "<< numSv[i] <<"\n";
    }

    out.close();

    return true;
}

/**
 * Write oriented points into XYZ file.
 */
template <typename T>
bool WriteXYZPoints(const std::string filename,
                    const Array<Point3D<T> >& points,
                    const Array<RVector3D>& normals) {
    assert(points.size() == normals.size());

    std::ofstream out(filename);
    if (!out) {
        LOG(INFO) << "Cannot open XYZ file '" << filename
                  << "' for writing.";
        return false;
    }

    out << std::setprecision(12);
    for (int i = 0; i < points.size(); ++i) {
        const Point3D<T>& p = points[i];
        out << p.x << " " << p.y << " " << p.z << " ";
        const RVector3D& v = normals[i];
        out << v.x << " " << v.y << " " << v.z << "\n";
    }

    out.close();

    return true;
}

} // namespace io
} // namespace geometry
} // namespace cl

#endif // GEOMETRY_IO_XYZ_IO_H_
