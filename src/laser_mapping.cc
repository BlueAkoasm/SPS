#include <tf/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <execution>
#include <fstream>
#include<math.h>

#include "laser_mapping.h"
#include "utils.h"
#include"optCeres.h"

namespace LIO {

const double fx = 617.971050917033;
const double fy = 616.445131524790;
const double cx = 327.710279392468;
const double cy = 253.976983707814;


bool LaserMapping::InitROS(ros::NodeHandle &nh) {
    LoadParams(nh);
    SubAndPubToROS(nh);

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        options::NUM_MAX_ITERATIONS, epsi.data());

    return true;
}

bool LaserMapping::InitWithoutROS(const std::string &config_yaml) {
    LOG(INFO) << "init laser mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        options::NUM_MAX_ITERATIONS, epsi.data());

    if (std::is_same<IVoxType, IVox<3, IVoxNodeType::PHC, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using phc ivox";
    } else if (std::is_same<IVoxType, IVox<3, IVoxNodeType::DEFAULT, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using default ivox";
    }

    return true;
}

bool LaserMapping::LoadParams(ros::NodeHandle &nh) {
    // get params from param server
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_surf_min;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;

    nh.param<bool>("path_save_en", path_save_en_, true);
    nh.param<bool>("publish/path_publish_en", path_pub_en_, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en_, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en_, false);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en_, true);
    nh.param<bool>("publish/scan_effect_pub_en", scan_effect_pub_en_, false);
    nh.param<std::string>("publish/tf_imu_frame", tf_imu_frame_, "body");
    nh.param<std::string>("publish/tf_world_frame", tf_world_frame_, "camera_init");

    nh.param<int>("max_iteration", options::NUM_MAX_ITERATIONS, 4);
    nh.param<float>("esti_plane_threshold", options::ESTI_PLANE_THRESHOLD, 0.1);
    nh.param<std::string>("map_file_path", map_file_path_, "");
    nh.param<bool>("common/time_sync_en", time_sync_en_, false);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min_, 0.0);
    nh.param<double>("MI_threshold", MI_threshold, 10.0);
    nh.param<bool>("MI_used", MI_used, false);
    nh.param<double>("MP_threshold", MP_threshold, 2.0);
    nh.param<double>("cube_side_length", cube_len_, 200);
    nh.param<float>("mapping/det_range", det_range_, 300.f);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", preprocess_->Blind(), 0.01);
    nh.param<float>("preprocess/time_scale", preprocess_->TimeScale(), 1e-3);
    nh.param<int>("preprocess/lidar_type", lidar_type, 1);
    nh.param<int>("preprocess/scan_line", preprocess_->NumScans(), 16);
    nh.param<int>("point_filter_num", preprocess_->PointFilterNum(), 2);
    nh.param<bool>("feature_extract_enable", preprocess_->FeatureEnabled(), false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log_, true);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en_, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en_, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval_, -1);
    nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT_, std::vector<double>());
    nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR_, std::vector<double>());

    nh.param<float>("ivox_grid_resolution", ivox_options_.resolution_, 0.2);
    nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = "camera_init";

    voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    return true;
}

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    // get params from yaml
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_surf_min;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;

    auto yaml = YAML::LoadFile(yaml_file);
    try {
        path_pub_en_ = yaml["publish"]["path_publish_en"].as<bool>();
        scan_pub_en_ = yaml["publish"]["scan_publish_en"].as<bool>();
        dense_pub_en_ = yaml["publish"]["dense_publish_en"].as<bool>();
        scan_body_pub_en_ = yaml["publish"]["scan_bodyframe_pub_en"].as<bool>();
        scan_effect_pub_en_ = yaml["publish"]["scan_effect_pub_en"].as<bool>();
        tf_imu_frame_ = yaml["publish"]["tf_imu_frame"].as<std::string>("body");
        tf_world_frame_ = yaml["publish"]["tf_world_frame"].as<std::string>("camera_init");
        path_save_en_ = yaml["path_save_en"].as<bool>();

        options::NUM_MAX_ITERATIONS = yaml["max_iteration"].as<int>();
        options::ESTI_PLANE_THRESHOLD = yaml["esti_plane_threshold"].as<float>();
        time_sync_en_ = yaml["common"]["time_sync_en"].as<bool>();

        filter_size_surf_min = yaml["filter_size_surf"].as<float>();
        filter_size_map_min_ = yaml["filter_size_map"].as<float>();
        MI_threshold = yaml["MI_threshold"].as<double>();
        MI_used = yaml["MI_used"].as<bool>();
        cube_len_ = yaml["cube_side_length"].as<int>();
        det_range_ = yaml["mapping"]["det_range"].as<float>();
        gyr_cov = yaml["mapping"]["gyr_cov"].as<float>();
        acc_cov = yaml["mapping"]["acc_cov"].as<float>();
        b_gyr_cov = yaml["mapping"]["b_gyr_cov"].as<float>();
        b_acc_cov = yaml["mapping"]["b_acc_cov"].as<float>();
        preprocess_->Blind() = yaml["preprocess"]["blind"].as<double>();
        preprocess_->TimeScale() = yaml["preprocess"]["time_scale"].as<double>();
        lidar_type = yaml["preprocess"]["lidar_type"].as<int>();
        preprocess_->NumScans() = yaml["preprocess"]["scan_line"].as<int>();
        preprocess_->PointFilterNum() = yaml["point_filter_num"].as<int>();
        preprocess_->FeatureEnabled() = yaml["feature_extract_enable"].as<bool>();
        extrinsic_est_en_ = yaml["mapping"]["extrinsic_est_en"].as<bool>();
        pcd_save_en_ = yaml["pcd_save"]["pcd_save_en"].as<bool>();
        pcd_save_interval_ = yaml["pcd_save"]["interval"].as<int>();
        extrinT_ = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

        ivox_options_.resolution_ = yaml["ivox_grid_resolution"].as<float>();
        ivox_nearby_type = yaml["ivox_nearby_type"].as<int>();
    } catch (...) {
        LOG(ERROR) << "bad conversion";
        return false;
    }

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    run_in_offline_ = true;
    return true;
}

void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    // ROS subscribe initialization
    std::string lidar_topic, imu_topic, image_topic;
    nh.param<std::string>("common/lid_topic", lidar_topic, "/livox/lidar");
    nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<std::string>("common/image_topic", image_topic, "/camera/color/image_raw");
    if (preprocess_->GetLidarType() == LidarType::AVIA) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            lidar_topic, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });
    // sub_img_ = nh.subscribe<sensor_msgs::CompressedImage>(
    // "/camera/color/image_raw/compressed", 10,
    // [this](const sensor_msgs::CompressedImageConstPtr &m){ this->ImageCallBack(m); });

    // ROS publisher init
    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = "camera_init";

    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_cloud_colored = nh.advertise<sensor_msgs::PointCloud2>("/cloud_colored", 100000);
    pub_laser_cloud_body_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
}

LaserMapping::LaserMapping() {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
    common::M3D R_CL;
    R_CL << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    common::V3D t_CL(0.30456, 0.00065, 0.65376);
    R_cl = R_CL.transpose();
    t_cl = -R_cl * t_CL;
}

void LaserMapping::Run() {
    if (!SyncPackages()) {
        return;
    }

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);
    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return;
    }
    //主要要改的内容
    /*
    1. 添加点的协方差计算
    2. 每个体素添加一个变量plane,  plane包含法向量n，和中心q，以及其协方差
    3. 观测模型
    4. 地图体素及平面拟合的更新
    
    难题：
    如何找到当前点的最近平面呢？
    */

    std::vector<PointWithCov> pv_vec;
    //计算雷达帧中每个点的协方差
    StatePose CurrentPose;
    /// the first scan
    if (flg_first_scan_) {
        //在一个体素中，如果点的数量大于5，那么就用信息最大的五个点预先拟合成一个平面作为这个体素的平面，并记录五个中的信息贡献最小值
        // 新加入的点会根据其所在体素，直接构成点到平面距离，无需寻找最近邻的点。 如果当前体素中没有拟合平面，那么寻找最近的点所在体素中有没有拟合平面
        // 如果周围的体素中点的数量都很少，都没有拟合平面，那么就按照原来的方法 选择最近的五个点进行拟合平面
        // 地图更新： 对于新加入到地图中的点，如果其信息贡献比最小值大了一个阈值，那么我们重新拟合平面
        // 其余的地图更新，还是按照之前的来
        // CreatePointCovVector(pv_vec, scan_undistort_);
        ivox_->AddPoints(scan_undistort_->points);
        // ivox_->AddPointsWithCov(pv_vec);
        first_lidar_time_ = measures_.lidar_bag_time_;
        flg_first_scan_ = false;
        // CurrentPose.Rotation = common::M3D::Identity();
        // CurrentPose.Position = common::V3D::Zero();
        // HistoricPoses.push_back(CurrentPose);
        return;
    }
    flg_EKF_inited_ = (measures_.lidar_bag_time_ - first_lidar_time_) >= options::INIT_TIME;

    /// downsample
    Timer::Evaluate(
        [&, this]() {
            voxel_scan_.setInputCloud(scan_undistort_);
            voxel_scan_.filter(*scan_down_body_);
            // scan_down_body_ = scan_undistort_;
        },
        "Downsample PointCloud");

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    cov_nearest.resize(cur_pts);
    residuals_.resize(cur_pts, 0);
    // residualsAvg_.resize(cur_pts, common::V3D::Zero());
    // residualsCov_.resize(cur_pts, common::M3D::Zero());
    // nearest_cov_.resize(cur_pts, common::M3D::Zero());
    // point_this_cov.resize(cur_pts, common::M3D::Zero());
    // nearest_pos_.resize(cur_pts, common::V3D::Zero());
    // point_this_pos_.resize(cur_pts, common::V3D::Zero());
    lowpoints.resize(cur_pts, true);
    // s_cov.resize(cur_pts, 0);

    point_selected_surf_.resize(cur_pts, true);
    plane_coef_.resize(cur_pts, common::V4F::Zero());
    highlight_index.clear();

    //检查每个体素是否要拟合平面

    // ICP and iterated Kalman filter update
    Timer::Evaluate(
        [&, this]() {
            // iterated state estimation
            double solve_H_time = 0;
            // update the observation model, will call nn and point-to-plane residual computation
            kf_.update_iterated_dyn_share_modified(options::LASER_POINT_COV, solve_H_time);
            // save the state
            state_point_ = kf_.get_x();
            euler_cur_ = SO3ToEuler(state_point_.rot);
            pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
            // CurrentPose.Rotation = state_point_.rot.toRotationMatrix().cast<double>();
            // CurrentPose.Position = state_point_.pos;
            // HistoricPoses.push_back(CurrentPose);
        },
        "IEKF Solve and Update");
    /*      用前n帧来重新优化一下当前帧位姿      */
    

    

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");

    // LOG(INFO)<<"IMU size is: "<<p_imu_->IMURelativePose_.size()<<"  Pose size is: "<<HistoricPoses.size();
    LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " downsamp " << cur_pts
              << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

    // publish or save map pcd
    if (run_in_offline_) {
        if (pcd_save_en_) {
            PublishFrameWorld();
        }
        if (path_save_en_) {
            PublishPath(pub_path_);
        }
    } else {
        if (pub_odom_aft_mapped_) {
            PublishOdometry(pub_odom_aft_mapped_);
        }
        if (path_pub_en_ || path_save_en_) {
            PublishPath(pub_path_);
        }
        if (scan_pub_en_ || pcd_save_en_) {
            PublishFrameWorld();
        }
        if (scan_pub_en_ && scan_body_pub_en_) {
            PublishFrameBody(pub_laser_cloud_body_);
        }
        if (scan_pub_en_ && scan_effect_pub_en_) {
            PublishFrameEffectWorld(pub_laser_cloud_effect_world_);
        }
    }

    // Debug variables
    frame_num_++;
}

void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.push_back(ptr);
            time_buffer_.push_back(msg->header.stamp.toSec());
            last_timestamp_lidar_ = msg->header.stamp.toSec();
        },
        "Preprocess (Standard)");
    mtx_buffer_.unlock();
}

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                LOG(WARNING) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            last_timestamp_lidar_ = msg->header.stamp.toSec();

            if (!time_sync_en_ && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
                !lidar_buffer_.empty()) {
                LOG(INFO) << "IMU and LiDAR not Synced, IMU time: " << last_timestamp_imu_
                          << ", lidar header time: " << last_timestamp_lidar_;
            }

            if (time_sync_en_ && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
                !imu_buffer_.empty()) {
                timediff_set_flg_ = true;
                timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
                LOG(INFO) << "Self sync IMU and LiDAR, time diff is " << timediff_lidar_wrt_imu_;
            }

            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.emplace_back(ptr);
            time_buffer_.emplace_back(last_timestamp_lidar_);
        },
        "Preprocess (Livox)");

    mtx_buffer_.unlock();
}

void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
    publish_count_++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu_) > 0.1 && time_sync_en_) {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu_ + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer_.lock();
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.emplace_back(msg);
    mtx_buffer_.unlock();
}

void LaserMapping::ImageCallBack(const sensor_msgs::CompressedImageConstPtr &msg)
{
    image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);

    if (image.empty()) { ROS_WARN("Empty image"); return; }
    // 用 img 做你的处理

}


bool LaserMapping::SyncPackages() {
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed_) {
        measures_.lidar_ = lidar_buffer_.front();
        measures_.lidar_bag_time_ = time_buffer_.front();

        if (measures_.lidar_->points.size() <= 1) {
            LOG(WARNING) << "Too few input point cloud!";
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
        } else if (measures_.lidar_->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_) {
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
        } else {
            scan_num_++;
            lidar_end_time_ = measures_.lidar_bag_time_ + measures_.lidar_->points.back().curvature / double(1000);
            lidar_mean_scantime_ +=
                (measures_.lidar_->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
        }

        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    double imu_time = imu_buffer_.front()->header.stamp.toSec();
    measures_.imu_.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        imu_time = imu_buffer_.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time_) break;
        measures_.imu_.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;
    return true;
}

void LaserMapping::PrintState(const state_ikfom &s) {
    LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose()
              << ", off r: " << s.offset_R_L_I.coeffs().transpose() << ", t: " << s.offset_T_L_I.transpose();
}

void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];
            auto &cov_near = cov_nearest[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            // double test = 0.0;
            // common::V3F p_body = scan_down_body_->points[i].getVector3fMap();
            // calcMI(p_body, cov_near, test);
            // if(test < MP_threshold) return;

            bool need_add = true;
            float dist = common::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= options::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                    if (common::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }
                   

            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}
void LaserMapping::MapIncrementalWithCov()
{
    // std::cout<<"----------------mapincremental-----------------"<<std::endl;
    // PointVector points_to_add;
    // PointVector point_no_need_downsample;
    std::vector<PointWithCov> points_to_add;
    std::vector<PointWithCov> point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        if(!lowpoints[i]) return;
        PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));
        /* decide if need add to map */
        PointWithCov pv;
        common::M3D cov;
        pv.point  = &scan_down_body_->points[i];
        common::V3D p_body(pv.point->x, pv.point->y, pv.point->z);
        calcBodyCov(p_body, 0.02, 0.05, cov);
        pv.cov = cov;
        common::M3D r_ = state_point_.rot.toRotationMatrix();
        common::M3D R_body =state_point_.offset_R_L_I.toRotationMatrix().cast<double>(); 
        pv.cov_world = (r_ * R_body * cov* R_body.transpose() * r_.transpose());
        pv.point_world = &scan_down_world_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((pv.point_world->getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                point_no_need_downsample.emplace_back(pv);
                return;
            }

            bool need_add = true;
            float dist = common::calc_dist(pv.point_world->getVector3fMap(), center);
            if (points_near.size() >= options::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                    if (common::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }
            if (need_add) {
                points_to_add.emplace_back(pv);
            }
        } else {
            points_to_add.emplace_back(pv);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPointsWithCov(points_to_add);
            ivox_->AddPointsWithCov(point_no_need_downsample);
        },
        "    IVox Add Points");    
}
/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * @param s kf state
 * @param ekfom_data H matrix
 */
void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    int cnt_pts = scan_down_body_->size();

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    //做特征筛选
    // 计算所有点的贡献， J^T *sigma *J (取前6乘6维) 
    // 先排序 再计算互信息值，不断加入最大的点 (100个)
    // 将这个点集中的点进行IEKF优化
    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
            auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

            // calLidarPointContribution(used);
            /** closest surface search and residual computation **/
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_->points[i];
                PointType &point_world = scan_down_world_->points[i];
                common::VV3D v_nearest;
                /* transform to world frame */
                common::V3F p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                auto &cov_near = cov_nearest[i];
                if (ekfom_data.converge) {
                    /** Find the closest surfaces in the map **/
                    ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                    point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;

                    /******************************lcx*********************************** */
                    if( MI_used && point_selected_surf_[i]){
                        // for(auto pn : points_near)
                        //     v_nearest.push_back(pn.getVector3fMap().cast<double>());
                        double test = 0.0;
                        // // if(cov_near.size() == 0) LOG(INFO)<<"cov nearest "<<v_nearest.size();
                        // calcMutualInformation(p_body, point_world.getVector3fMap(), v_nearest, cov_near, i, test);
                        calcKL(p_body, points_near,cov_near, test );
                        // LOG(INFO)<<"test: "<<test;
                        if(test > MI_threshold)
                         {
                                highlight_index.insert(i);
                               point_selected_surf_[i] = false;
                         }
                    }
                    /******************************lcx*********************************** */

                    if (point_selected_surf_[i]) {
                        point_selected_surf_[i] =
                            common::esti_plane(plane_coef_[i], points_near, options::ESTI_PLANE_THRESHOLD);
                    }
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);


                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    }
                }

            });
        },
        "    ObsModel (Lidar Match)");

    effect_feat_num_ = 0;

    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    cur_cov_.resize(cnt_pts);
    s_cov_final.resize(cnt_pts);

    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];
            // residualsAvg_[effect_feat_num_] = point_this_pos_[i]-nearest_pos_[i];
            // residualsCov_[effect_feat_num_] =  ( point_this_cov[i] + nearest_cov_[i] + 1e-3*common::M3D::Identity()).inverse();
            // s_cov_final[effect_feat_num_] = s_cov[i];
            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);
    // residualsCov_.resize(effect_feat_num_);
    // residualsAvg_.resize(effect_feat_num_);
    // cur_cov_.resize(effect_feat_num_);
    // s_cov_final.resize(effect_feat_num_);


    if (effect_feat_num_ < 1) {
        ekfom_data.valid = false;
        LOG(WARNING) << "No Effective Points!";
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            ekfom_data.h.resize(effect_feat_num_);
            ekfom_data.h_cov_x = Eigen::MatrixXd::Zero(3*effect_feat_num_, 12);
            ekfom_data.h_cov.resize(3*effect_feat_num_);

            index.resize(effect_feat_num_);
            const common::M3F off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
            const common::V3F off_t = s.offset_T_L_I.cast<float>();
            const common::M3F Rt = s.rot.toRotationMatrix().transpose().cast<float>();
            const common::M3F R_ = Rt.transpose();
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                common::V3F point_this_be = corr_pts_[i].head<3>();
                common::M3F point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                common::V3F point_this = off_R * point_this_be + off_t;
                common::M3F point_crossmat = SKEW_SYM_MATRIX(point_this);

                /*** get the normal vector of closest surface/corner ***/
                common::V3F norm_vec = corr_norm_[i].head<3>();

                /*** calculate the Measurement Jacobian matrix H ***/
                common::V3F C(Rt * norm_vec);
                common::V3F A(point_crossmat * C);

                if (extrinsic_est_en_) {
                    common::V3F B(point_be_crossmat * off_R.transpose() * C);
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                        B[1], B[2], C[0], C[1], C[2];
                } else {
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0;
                }

                
                

                /*** Measurement: distance to the closest surface/corner ***/
                ekfom_data.h(i) = -corr_pts_[i][3];
                
                // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(residualsCov_[i]);
                // Eigen::Matrix3d U = solver.eigenvectors();

                // Eigen::Matrix3d Q = Eigen::Matrix3d::Identity();

                
                // ekfom_data.h_cov.block<3,1>(3*i,0) = - s_cov_final[i] * Q * U * residualsAvg_[i];
                // // ekfom_data.h_cov_x.block<3,12>(3*i,0) = Eigen::Matrix<double, 3, 12>::Zero();
                // common::V3F  tmp = R_*point_this;
                // common::M3F D =   point_crossmat *Rt;
                // ekfom_data.h_cov_x.block<3,3>(3*i,3) = s_cov_final[i] * Q * U *D.cast<double>();
                
                // ekfom_data.h_cov_x.block<3,3>(3*i,0) =  s_cov_final[i] * Q * U;

                
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
}

/////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

pcl::PointCloud<pcl::PointXYZRGB>::Ptr LaserMapping::ColoredPointCloud(PointCloudType::Ptr& in,PointCloudType::Ptr& body, std::unordered_set<size_t> highlight_idx)
{
    auto out = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    out->points.resize(in->points.size());
    out->width  = static_cast<uint32_t>(in->points.size());
    out->height = 1;
    out->is_dense = in->is_dense;

    // 遍历每个点
    for (size_t i = 0; i < in->points.size(); ++i) {
        if(!highlight_idx.count(i)) continue;
        const auto& src = in->points[i];
        const auto& src2 = body->points[i];
        auto& dst = out->points[i];

        common::V3D pL(src2.x, src2.y, src2.z);
        common::V3D pC = R_cl * pL + t_cl;
        if (pC.z() <= 0.0) continue;

    // 3) Pinhole 投影（无畸变）
        double invZ = 1.0 / pC.z();
        double u = fx * (pC.x() * invZ) + cx;
        double v = fy * (pC.y() * invZ) + cy;

        // 4) 在图像范围内再绘制/使用
        if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
            // 画一个小点（或做其他叠加）
            cv::circle(image, cv::Point(static_cast<int>(u), static_cast<int>(v)),
                    3, cv::Scalar(0,0,255), -1, cv::LINE_AA);
        }
        
        // 坐标
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;

        dst.r = 255;
        dst.g = 0;
        dst.b = 0;

    }
    cv::imshow("im", image);
    cv::waitKey(1);
    return out;
}



void LaserMapping::PublishPath(const ros::Publisher pub_path) {
    SetPosestamp(msg_body_pose_);
    msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time_);
    msg_body_pose_.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    path_.poses.push_back(msg_body_pose_);
    if (run_in_offline_ == false) {
        pub_path.publish(path_);
    }
}

void LaserMapping::PublishOdometry(const ros::Publisher &pub_odom_aft_mapped) {
    odom_aft_mapped_.header.frame_id = "camera_init";
    odom_aft_mapped_.child_frame_id = "body";
    odom_aft_mapped_.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
    SetPosestamp(odom_aft_mapped_.pose);
    pub_odom_aft_mapped.publish(odom_aft_mapped_);
    auto P = kf_.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odom_aft_mapped_.pose.covariance[i * 6 + 0] = P(k, 3);
        odom_aft_mapped_.pose.covariance[i * 6 + 1] = P(k, 4);
        odom_aft_mapped_.pose.covariance[i * 6 + 2] = P(k, 5);
        odom_aft_mapped_.pose.covariance[i * 6 + 3] = P(k, 0);
        odom_aft_mapped_.pose.covariance[i * 6 + 4] = P(k, 1);
        odom_aft_mapped_.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odom_aft_mapped_.pose.pose.position.x, odom_aft_mapped_.pose.pose.position.y,
                                    odom_aft_mapped_.pose.pose.position.z));
    q.setW(odom_aft_mapped_.pose.pose.orientation.w);
    q.setX(odom_aft_mapped_.pose.pose.orientation.x);
    q.setY(odom_aft_mapped_.pose.pose.orientation.y);
    q.setZ(odom_aft_mapped_.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odom_aft_mapped_.header.stamp, tf_world_frame_, tf_imu_frame_));
}

void LaserMapping::PublishFrameWorld() {
    if (!(run_in_offline_ == false && scan_pub_en_) && !pcd_save_en_) {
        return;
    }

    PointCloudType::Ptr laserCloudWorld;
    
    if (dense_pub_en_) {
        PointCloudType::Ptr laserCloudFullRes(scan_undistort_);
        int size = laserCloudFullRes->points.size();
        laserCloudWorld.reset(new PointCloudType(size, 1));
        for (int i = 0; i < size; i++) {
            PointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }
    } else {
        laserCloudWorld = scan_down_world_;
    }

    if (run_in_offline_ == false && scan_pub_en_) {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudmsg.header.frame_id = "camera_init";
        pub_laser_cloud_world_.publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;

        // auto colored = ColoredPointCloud(scan_down_world_, scan_down_body_,highlight_index);
        // sensor_msgs::PointCloud2 msg_rgb;
        // pcl::toROSMsg(*colored, msg_rgb);
        // msg_rgb.header.stamp    = ros::Time().fromSec(lidar_end_time_);    // 与原始保持一致
        // msg_rgb.header.frame_id = "camera_init"; // 与原始保持一致
        // pub_cloud_colored.publish(msg_rgb);
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en_) {
        *pcl_wait_save_ += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save_->size() > 0 && pcd_save_interval_ > 0 && scan_wait_num >= pcd_save_interval_) {
            pcd_index_++;
            std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/scans_") + std::to_string(pcd_index_) +
                                       std::string(".pcd"));
            pcl::PCDWriter pcd_writer;
            LOG(INFO) << "current scan saved to /PCD/" << all_points_dir;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
            pcl_wait_save_->clear();
            scan_wait_num = 0;
        }
    }
}

void LaserMapping::PublishFrameBody(const ros::Publisher &pub_laser_cloud_body) {
    int size = scan_undistort_->points.size();
    PointCloudType::Ptr laser_cloud_imu_body(new PointCloudType(size, 1));

    for (int i = 0; i < size; i++) {
        PointBodyLidarToIMU(&scan_undistort_->points[i], &laser_cloud_imu_body->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud_imu_body, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "body";
    pub_laser_cloud_body.publish(laserCloudmsg);
    publish_count_ -= options::PUBFRAME_PERIOD;
}

void LaserMapping::PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world) {
    int size = corr_pts_.size();
    PointCloudType::Ptr laser_cloud(new PointCloudType(size, 1));

    for (int i = 0; i < size; i++) {
        PointBodyToWorld(corr_pts_[i].head<3>(), &laser_cloud->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "camera_init";
    pub_laser_cloud_effect_world.publish(laserCloudmsg);
    publish_count_ -= options::PUBFRAME_PERIOD;
}

void LaserMapping::Savetrajectory(const std::string &traj_file) {
    std::ofstream ofs;
    ofs.open(traj_file, std::ios::out);
    if (!ofs.is_open()) {
        LOG(ERROR) << "Failed to open traj_file: " << traj_file;
        return;
    }

    for (const auto &p : path_.poses) {
        ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
            << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
            << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
    }
    Eigen::Vector3f a(path_.poses.back().pose.position.x, path_.poses.back().pose.position.y, path_.poses.back().pose.position.z);
    Eigen::Vector3f b(path_.poses.front().pose.position.x, path_.poses.front().pose.position.y, path_.poses.front().pose.position.z);
    std::cout<<"end-to-end drift: "<<(a-b).norm()<<std::endl;   
    ofs.close();
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////
template <typename T>
void LaserMapping::SetPosestamp(T &out) {
    out.pose.position.x = state_point_.pos(0);
    out.pose.position.y = state_point_.pos(1);
    out.pose.position.z = state_point_.pos(2);
    out.pose.orientation.x = state_point_.rot.coeffs()[0];
    out.pose.orientation.y = state_point_.rot.coeffs()[1];
    out.pose.orientation.z = state_point_.rot.coeffs()[2];
    out.pose.orientation.w = state_point_.rot.coeffs()[3];
}

void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po) {
    common::V3D p_body(pi->x, pi->y, pi->z);
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyToWorld(const common::V3F &pi, PointType *const po) {
    common::V3D p_body(pi.x(), pi.y(), pi.z());
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = std::abs(po->z);
}

void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    common::V3D p_body_lidar(pi->x, pi->y, pi->z);
    common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void LaserMapping::Finish() {
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save_->size() > 0 && pcd_save_en_) {
        std::string file_name = std::string("scans.pcd");
        std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        LOG(INFO) << "current scan saved to /PCD/" << file_name;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
    }

    LOG(INFO) << "finish done";
}
void LaserMapping::CreatePointCovVector(std::vector<PointWithCov>& pv_vec, CloudPtr scan_)
{
    for(size_t i=0 ; i<scan_->size(); i++)
    {
        PointWithCov  pv;
        pcl::PointXYZINormal point_copy = scan_->points[i];
        pv.point  = &point_copy;
        common::V3D p_body(pv.point->x, pv.point->y, pv.point->z);
        common::V3D p_global;
        p_global = kf_.get_x().rot * (kf_.get_x().offset_R_L_I * p_body + kf_.get_x().offset_T_L_I) + kf_.get_x().pos;

        pv.point_world->x = p_global(0);
        pv.point_world->y = p_global(1);
        pv.point_world->z = p_global(2);
        pv.point_world->intensity = pv.point->intensity;

        common::M3D cov;
        calcBodyCov(p_body, 0.02, 0.05, cov); //计算雷达坐标系的cov
        common::M3D R = kf_.get_x().rot.toRotationMatrix().cast<double>();
        common::M3D R_body =kf_.get_x().offset_R_L_I.toRotationMatrix().cast<double>();  //根据当前的状态计算世界坐标系下的cov
        pv.cov = cov;
        pv.cov_world = (R * R_body * cov *R_body.transpose()* R.transpose());
        pv_vec.push_back(pv);
        // std::cout<<pv.cov_world<<std::endl;
    }
}


void LaserMapping::calcBodyCov(common::V3D &pb, const float range_inc, const float degree_inc, common::M3D &cov)
{
    float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
    float range_var = range_inc * range_inc;
    Eigen::Matrix2d direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
        pow(sin(DEG2RAD(degree_inc)), 2);
    Eigen::Vector3d direction(pb);
    direction.normalize();
    Eigen::Matrix3d direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0,
        -direction(0), -direction(1), direction(0), 0;
    Eigen::Vector3d base_vector1(1, 1,
                                 -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
        base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    common::M3D cov_ = direction * range_var * direction.transpose() +
          A * direction_var * A.transpose();   
    // std::cout<<"cov is"<<cov<<std::endl;
    cov = 1e3*cov_;
    // LOG(INFO) << "cov "<<cov;
    
    return;
}

void LaserMapping::CalcWorldCov(PointWithCov& pv, common::M3D rot)
{
    common::V3D p_body(pv.point->x, pv.point->y, pv.point->z);
    common::M3D point_crossmat = SKEW_SYM_MATRIX(p_body);

    pv.cov_world = (rot * pv.cov * rot.transpose());
    return;
}

static bool custom_compare(std::pair<PointWithCov, int>&a, std::pair<PointWithCov, int>& b)
{
    return a.first.contribution.trace() > b.first.contribution.trace();
}

void LaserMapping::calLidarPointContribution(std::vector<bool>& used)
{
    std::vector<PointWithCov> pv_vec;
    std::vector<std::pair<PointWithCov, int>> indexofp;
    CreatePointCovVector(pv_vec, scan_down_body_);
    // std::cout<<pv_vec.size()<<std::endl;
    indexofp.resize(pv_vec.size());
    used.resize(pv_vec.size());
    const common::M3D off_R = kf_.get_x().offset_R_L_I.toRotationMatrix().cast<double>();
    const common::V3D off_t = kf_.get_x().offset_T_L_I.cast<double>();
    for(int i=0; i<pv_vec.size(); i++)
    {
        PointWithCov pv = pv_vec[i];
        Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
        common::V3D p_lidar = (pv.point->getVector3fMap()).cast<double>();
        common::V3D p_body = off_R * p_lidar + off_t;
        
        auto R_wb = (kf_.get_x().rot).cast<double>();

        common::M3D J1 = -SKEW_SYM_MATRIX(R_wb * p_body);

        J.block<3,3>(0,0) = J1;

        J.block<3,3>(0,3) = common::M3D::Identity();
        pv.contribution = J.transpose() * pv.cov.inverse() * J;

        indexofp[i] = std::make_pair(pv, i);
        
    }
    std::sort(indexofp.begin(), indexofp.end(), custom_compare);
    int k = indexofp.size()/2;
    
    for(int j = 0; j<k ;j++)
    {
        used[indexofp[j].second] = true;
    }
}
void LaserMapping::calcMutualInformation(common::V3F p_body, common::V3F point_world, 
    common::VV3D& near_point, common::VM3D& near_points_cov, int index,
    double &result) {
// 默认结果为0（异常时直接返回）
result = MI_threshold+1;

// 检查输入有效性
if (near_point.empty() || near_points_cov.empty() || near_point.size() != near_points_cov.size()) {
return;
}


common::M3D cov;
common::V3D cur_p = p_body.cast<double>();
calcBodyCov(cur_p, 0.02, 0.05, cov);
common::M3D R = kf_.get_x().rot.toRotationMatrix().cast<double>();
common::M3D R_body =kf_.get_x().offset_R_L_I.toRotationMatrix().cast<double>(); 
common::M3D sigma1 = R * R_body * cov* R_body.transpose() * R.transpose();
common::V3D mu1 = point_world.cast<double>();

// 计算近邻点的均值和协方差
common::V3D mu2 = common::V3D::Zero();
common::M3D sigma2 = common::M3D::Zero();

for (size_t i = 0; i < near_point.size(); i++) {
mu2 += near_point[i];
sigma2 += near_points_cov[i];
}
mu2 /= near_point.size(); 
sigma2 /= near_points_cov.size();

// --- 针对小行列式的阈值调整 ---
const double kCovDetThreshold = 1e-13;  
const double kSVDRatioThreshold = 1e-3; 


// 检查协方差矩阵是否有效
double sig1_det = sigma1.determinant();
double sig2_det = sigma2.determinant();
if (std::abs(sig1_det) < kCovDetThreshold || std::abs(sig2_det) < kCovDetThreshold) {
    LOG(INFO) <<" inverse is invaild ";
return;
}

// --- 使用更宽松的SVD求逆 ---
Eigen::JacobiSVD<common::M3D> svd(sigma2, Eigen::ComputeFullU | Eigen::ComputeFullV);
const auto &sing_vals = svd.singularValues();

// 检查最小奇异值与最大奇异值的比例（更宽松的阈值）
double max_singular = sing_vals.maxCoeff();
double min_singular = sing_vals.minCoeff();
if (min_singular < kSVDRatioThreshold * max_singular) {
    LOG(INFO) <<" coeff is invaild "; // 原阈值比例：1e-6 → 1e-3
return;
}

// 显式构造伪逆（避免直接求逆的数值问题）
common::M3D sigma2_inv = svd.matrixV() * 
(sing_vals.array().inverse().matrix().asDiagonal()) * 
svd.matrixU().adjoint();

// 计算互信息项
double trace_term = (sigma2_inv * sigma1).trace();
common::V3D diff = mu2 - mu1;
double diff_term = diff.transpose()  *sigma2_inv * diff;
    // 新增的log项：log(sigma2的行列式 / sigma1的行列式)
double log_term = std::log(sig2_det) - std::log(sig1_det); // 等价于 log(sig2_det/sig1_det)
double a = trace_term + diff_term + log_term-3;
// LOG(INFO)<<"trace term: "<<trace_term<<" diff_term: "<<diff_term<<" long_term: "<<log_term<<"  result :"<<a;
// 确保结果有效后再赋值
if (std::isfinite(a)) {
    result = a;
    // nearest_cov_[index] = sigma2;
    // point_this_cov[index] = sigma1;
    // point_this_pos_[index] = mu1;
    // nearest_pos_[index] = mu2;
    // double sig1_det_2 = (sigma1/2).determinant();
    // double sig2_det_2 = (sigma2/2).determinant();
    // s_cov[index] = std::sqrt(std::sqrt(sig2_det*sig1_det)/(sig1_det_2+sig2_det_2));
}
}
void LaserMapping::calcKL(common::V3F p_body, PointVector near_points, common::VM3D& cov_near, double& result)
{
    auto R_L_I = kf_.get_x().offset_R_L_I.cast<double>();
    auto T_L_I = kf_.get_x().offset_T_L_I.cast<double>();
    auto R_L_W = kf_.get_x().rot.toRotationMatrix().inverse();
    auto T_L_W = -R_L_W * kf_.get_x().pos;
    common::VM3D near_cov;
    common::M3D cur_cov;

    common::V3D mu1 = p_body.cast<double>();
    common::V3D mu2 = common::V3D::Zero();

    common::M3D sigma1;
    common::M3D sigma2 = common::M3D::Zero();
    mu1 = R_L_I * mu1 + T_L_I;
    calcBodyCov(mu1, 0.02, 0.05, sigma1);

    for(auto& p : near_points)
    {
        common::V3D p_w = p.getVector3fMap().cast<double>();
        common::V3D p_L = R_L_W.cast<double>() * p_w + T_L_W.cast<double>();
        common::M3D sig;
        mu2 += p_L;
        calcBodyCov(p_L, 0.02, 0.05, sig);
        sigma2 += sig;
        cov_near.push_back(sig);

    }
    mu2 = mu2 / near_points.size();
    sigma2 = sigma2 /  near_points.size();

    const double kCovDetThreshold = 1e-13;  
    const double kSVDRatioThreshold = 1e-6; 
    
    
    // 检查协方差矩阵是否有效
    double sig1_det = sigma1.determinant();
    double sig2_det = sigma2.determinant();
    if (std::abs(sig1_det) < kCovDetThreshold || std::abs(sig2_det) < kCovDetThreshold) {
        // LOG(INFO) <<" inverse is invaild ";
    return;
    }
    
    // --- 使用更宽松的SVD求逆 ---
    Eigen::JacobiSVD<common::M3D> svd(sigma2, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &sing_vals = svd.singularValues();
    
    // 检查最小奇异值与最大奇异值的比例（更宽松的阈值）
    double max_singular = sing_vals.maxCoeff();
    double min_singular = sing_vals.minCoeff();
    if (min_singular < kSVDRatioThreshold * max_singular) {
        // LOG(INFO) <<" coeff is invaild "; // 原阈值比例：1e-6 → 1e-3
    return;
    }
    
    // 显式构造伪逆（避免直接求逆的数值问题）
    common::M3D sigma2_inv = svd.matrixV() * 
    (sing_vals.array().inverse().matrix().asDiagonal()) * 
    svd.matrixU().adjoint();
    
    // 计算互信息项
    double trace_term = (sigma2_inv * sigma1).trace();
    common::V3D diff = mu2 - mu1;
    double diff_term = diff.transpose()  *sigma2_inv * diff;
        // 新增的log项：log(sigma2的行列式 / sigma1的行列式)
    double log_term = std::log(sig2_det) - std::log(sig1_det); // 等价于 log(sig2_det/sig1_det)
    double a = trace_term + diff_term + log_term-3;
    // LOG(INFO)<<"trace term: "<<trace_term<<"  diff_term: "<<diff_term<<"  long_term: "<<log_term<<"  result :"<<a;
    // 确保结果有效后再赋值
    if (std::isfinite(a)) {
        result = a;
    }

}
void LaserMapping::calcMI(common::V3F p_body, common::VM3D& cov_near, double& result)
{
    // 使用 const 引用避免拷贝
    const auto& R_L_I = kf_.get_x().offset_R_L_I.cast<double>();
    const auto& T_L_I = kf_.get_x().offset_T_L_I.cast<double>();

    // 坐标变换（优化了中间变量）
    common::V3D mu1 = (R_L_I * p_body.cast<double>()) + T_L_I;

    // 计算当前点协方差
    common::M3D sigma1;
    calcBodyCov(mu1, 0.02, 0.05, sigma1);
    
    // 预分配并并行累加协方差矩阵（假设OpenMP可用）
    common::M3D sigma2 = common::M3D::Zero();
    for(size_t i = 0; i < cov_near.size(); ++i) {
        sigma2 += cov_near[i];
    }

    // 使用别名避免重复计算
    const common::M3D& sigma_total = sigma2 + sigma1;

    // 优先尝试Cholesky分解（适合对称正定矩阵）
    Eigen::LLT<common::M3D> llt2(sigma2);
    Eigen::LLT<common::M3D> llt_total(sigma_total);
    
    if (llt2.info() == Eigen::Success && llt_total.info() == Eigen::Success) {
        // 直接从Cholesky分解结果计算行列式（对角线元素乘积）
        Eigen::Matrix3d L2 = llt2.matrixL();
        Eigen::Matrix3d L_total = llt_total.matrixL();

        const double det2 = L2.diagonal().prod();
        const double det_total = L_total.diagonal().prod();

        result = std::log2(det_total / det2);

        return;
    }

    // 回退到优化的QR分解（使用固定大小矩阵避免动态分配）
    const auto qr2 = sigma2.householderQr();
    const auto qr_total = sigma_total.householderQr();
    
    // 直接计算上三角矩阵对角线乘积（比determinant()更快）
    const double det2 = qr2.matrixQR().diagonal().prod();
    const double det_total = qr_total.matrixQR().diagonal().prod();
    
    // 添加数值稳定性处理（防止log2负数）
    result = std::log2(std::abs(det_total) / std::abs(det2));
}
// void LaserMapping::calcMI(common::V3F p_body, common::VM3D& cov_near, double& result)
// {
//     auto R_L_I = kf_.get_x().offset_R_L_I.cast<double>();
//     auto T_L_I = kf_.get_x().offset_T_L_I.cast<double>();

//     common::VM3D near_cov;
//     common::M3D cur_cov;

//     common::V3D mu1 = p_body.cast<double>();
//     common::V3D mu2 = common::V3D::Zero();

//     common::M3D sigma1;
//     common::M3D sigma2 = common::M3D::Zero();
//     mu1 = R_L_I * mu1 + T_L_I;
//     calcBodyCov(mu1, 0.02, 0.05, sigma1);   
    
//     for(auto ma : cov_near)
//     {
//         sigma2 += ma;
//     }
//     Eigen::HouseholderQR<Eigen::MatrixXd> qr;
//     qr.compute(sigma2);
//     Eigen::MatrixXd R2 = qr.matrixQR().triangularView<Eigen::Upper>();

//     qr.compute(sigma2 + sigma1);
//     Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

//     double MI = std::log2(R.determinant()/R2.determinant());
//     result = MI;
// }


}  // namespace LIO
