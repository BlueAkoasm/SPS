#include<ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "/home/zxl2/LCX/project_lcx/LIO/src/LIO/thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "/home/zxl2/LCX/project_lcx/LIO/src/LIO/thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "/home/zxl2/LCX/project_lcx/LIO/src/LIO/thirdparty/g2o/g2o/core/block_solver.h"
#include "/home/zxl2/LCX/project_lcx/LIO/src/LIO/thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "/home/zxl2/LCX/project_lcx/LIO/src/LIO/thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "/home/zxl2/LCX/project_lcx/LIO/src/LIO/thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "/home/zxl2/LCX/project_lcx/LIO/src/LIO/thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "/home/zxl2/LCX/project_lcx/LIO/src/LIO/thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "/home/zxl2/LCX/project_lcx/LIO/src/LIO/thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include<laser_mapping.h>

using namespace LIO;
typedef g2o::VertexSim3Expmap VertexSim3;
typedef g2o::EdgeSim3 EdgeSim3;
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w) {
    Eigen::AngleAxisd aa(w.norm(), w.normalized());
    return aa.toRotationMatrix();
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}

Eigen::Matrix3d JacobianRInv(const Eigen::Vector3d& w) {
    const double theta = w.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity();
    }
    const Eigen::Matrix3d W = SKEW_SYM_MATRIX(w);
    return Eigen::Matrix3d::Identity() + 0.5 * W +
           (1.0 / (theta * theta) - (1 + cos(theta)) / (2 * theta * sin(theta))) * W * W;
}
void EKFUpdate(const std::vector<IMURelativeDelta>& imu_deltas,
    const std::vector<StatePose>& historic_poses,
    const SO3 R_BL,   // 雷达到IMU的旋转外参
    const Eigen::Vector3d t_BL, // 雷达到IMU的平移外参
    StatePose& current_pose,    // T_WL2 (优化变量)
    Eigen::Matrix<double, 6, 6>& P) {
const int n_residuals = 5;
Eigen::VectorXd z(6 * n_residuals);
Eigen::MatrixXd H(6 * n_residuals, 6);
Eigen::Matrix3d R_B1B2 = Eigen::Matrix3d::Identity();
Eigen::Vector3d t_B1B2 = Eigen::Vector3d::Zero();
for (int i = 0; i < n_residuals; ++i) {
    const int idx = historic_poses.size() - n_residuals - 1 + i;
    R_B1B2 *= imu_deltas[idx].delta_R;
    t_B1B2 += imu_deltas[idx].delta_t;
}
for (int i = 0; i < n_residuals; ++i) {
const int idx = historic_poses.size() - n_residuals - 1 + i;
const auto& hist_pose = historic_poses[idx];  // T_WL1

// 从IMU预积分获取 T_B1B2
if(i !=0 )
{
    R_B1B2 = imu_deltas[idx-1].delta_R.inverse() * R_B1B2;
    t_B1B2 -= imu_deltas[idx-1].delta_t;
}

// 计算 T_L1L2 = T_BL^{-1} * T_B1B2 * T_BL
Eigen::Matrix3d R_L1L2 = R_BL.inverse().matrix() * R_B1B2 * R_BL.matrix();
Eigen::Vector3d t_L1L2 = R_BL.inverse() * (R_B1B2 * t_BL + t_B1B2 - t_BL);

// 计算旋转残差 r_R = Log(R_L1L2 * R_WL2^T * R_WL1)
Eigen::Matrix3d R_err = R_L1L2 * current_pose.Rotation.transpose() * hist_pose.Rotation;
Eigen::Vector3d r_R = LogSO3(R_err);

// 计算平移残差 r_t = R_L1L2(-R_WL2^T t_WL2 + R_WL2^T t_WL1) + t_L1L2
Eigen::Vector3d t_err = -current_pose.Rotation.transpose() * current_pose.Position + 
                     current_pose.Rotation.transpose() * hist_pose.Position;
Eigen::Vector3d r_t = R_L1L2 * t_err + t_L1L2;

// 填充观测向量
z.segment<3>(6 * i) = r_R;
z.segment<3>(6 * i + 3) = r_t;

// 计算雅可比矩阵
Eigen::Matrix3d J_r_inv = JacobianRInv(r_R);
Eigen::Matrix<double, 6, 6> Hi = Eigen::Matrix<double, 6, 6>::Zero();

// 旋转残差对T_WL2旋转部分的导数
Hi.block<3, 3>(0, 0) = -J_r_inv * current_pose.Rotation.transpose();

// 平移残差对T_WL2旋转部分的导数
Eigen::Vector3d t_diff = hist_pose.Position - current_pose.Position;
Hi.block<3, 3>(3, 0) = -R_L1L2 * current_pose.Rotation.transpose() * SKEW_SYM_MATRIX(t_diff);

// 平移残差对T_WL2平移部分的导数
Hi.block<3, 3>(3, 3) = -R_L1L2 * current_pose.Rotation.transpose();

H.block<6, 6>(6 * i, 0) = Hi;
}

// 卡尔曼增益和状态更新
Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6 * n_residuals, 6 * n_residuals) * 0.01;
Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
Eigen::VectorXd dx = K * z;

// 更新位姿 (T_WL2)
current_pose.Rotation = current_pose.Rotation * ExpSO3(dx.head<3>());
current_pose.Position += dx.tail<3>();

// 协方差更新 (Joseph形式)
Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(6, 6) - K * H;
P = I_KH * P * I_KH.transpose() + K * R * K.transpose();
}
void IEKFUpdate(const std::vector<IMURelativeDelta>& imu_deltas,
    const std::vector<StatePose>& historic_poses,
    StatePose& current_pose,
    Eigen::Matrix<double, 6, 6>& P) {
// IEKF参数配置
    const int max_iterations = 5;        // 最大迭代次数
    const double convergence_thresh = 1e-4; // 收敛阈值（状态增量范数）
    StatePose initial_pose = current_pose; // 保存初始状态
    Eigen::Matrix<double, 6, 6> P_initial = P;

    for (int iter = 0; iter < max_iterations; ++iter) {
    // ------------------------------
    // 1. 构建观测向量z和雅可比矩阵H
    // ------------------------------
        const int n_residuals = 5; // 前5到前1帧共5个残差
        Eigen::VectorXd z(6 * n_residuals);
        Eigen::MatrixXd H(6 * n_residuals, 6);

        // 初始化IMU累积变换（每次迭代重置）
        Eigen::Matrix3d R_IMU = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_IMU = Eigen::Vector3d::Zero();

        for (int i = 0; i < n_residuals; ++i) {
        const int idx = historic_poses.size() - n_residuals - 1 + i;
        const auto& hist_pose = historic_poses[idx];

        // 累积IMU变换（注意：需根据当前迭代状态重新积分）
        R_IMU *= imu_deltas[idx].delta_R;
        t_IMU += imu_deltas[idx].delta_t;

        // 计算旋转残差r_Ri（基于当前迭代状态）
        Eigen::Matrix3d R_est = current_pose.Rotation * hist_pose.Rotation.inverse();
        Eigen::Vector3d r_R = LogSO3(R_IMU * R_est.transpose());

        // 计算平移残差r_ti（基于当前迭代状态）
        Eigen::Vector3d r_t = current_pose.Position - hist_pose.Position -
                            hist_pose.Rotation * t_IMU;

        // 填充观测向量z
        z.segment<3>(6 * i) = r_R;
        z.segment<3>(6 * i + 3) = r_t;

        // 计算雅可比矩阵Hi（基于当前迭代状态）
        Eigen::Matrix3d J_r_inv = JacobianRInv(r_R);
        Eigen::Matrix<double, 6, 6> Hi = Eigen::Matrix<double, 6, 6>::Zero();
        Hi.block<3, 3>(0, 0) = -J_r_inv * current_pose.Rotation.transpose();
        Hi.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();

        H.block<6, 6>(6 * i, 0) = Hi;
        }

        // ------------------------------
        // 2. 计算卡尔曼增益K
        // ------------------------------
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6 * n_residuals, 6 * n_residuals) * 0.1;
        Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

        // ------------------------------
        // 3. 计算状态增量dx
        // ------------------------------
        Eigen::VectorXd dx = K * z;

        // ------------------------------
        // 4. 更新状态（临时变量）
        // ------------------------------
        StatePose new_pose = current_pose;
        new_pose.Rotation = current_pose.Rotation * ExpSO3(dx.head<3>());
        new_pose.Position += dx.tail<3>();

        // ------------------------------
        // 5. 检查收敛条件
        // ------------------------------
        double delta_norm = dx.norm();
        if (delta_norm < convergence_thresh) {
        current_pose = new_pose;
        break; // 收敛则退出迭代
    }

    // ------------------------------
    // 6. 未收敛时更新状态和协方差
    // ------------------------------
    current_pose = new_pose;
    P = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P;

    // 若达到最大迭代次数，恢复初始协方差
    if (iter == max_iterations - 1) {
    P = P_initial;
}
}
}

void optimizeCurrentPoseWithG2O(const std::vector<IMURelativeDelta>& imu_deltas, const std::vector<StatePose>& historic_poses, 
    StatePose& current_pose, Eigen::Matrix3d R_L_I, Eigen::Vector3d t_L_I) {
    // ------------------------------
    // 1. 初始化优化器
    // ------------------------------
    Eigen::Matrix3d R_I_L = R_L_I.transpose();
    Eigen::Vector3d t_I_L = -t_L_I;
    g2o::Sim3 S_I_L(R_I_L, t_I_L, 1.0);
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);
    std::vector<g2o::Sim3> Sij(5);
    std::vector<g2o::Sim3> Sbibj(5);
    int k = 0;
    Eigen::Matrix3d R_IMU = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_IMU = Eigen::Vector3d::Zero();
    int n = historic_poses.size();
    for (size_t i = n-2; i >= n-6; --i) {
        R_IMU *= imu_deltas[i].delta_R;
        t_IMU += imu_deltas[i].delta_t;
        Sbibj[k++] = g2o::Sim3(R_IMU, t_IMU, 1);
    }
    // ------------------------------
    // 2. 添加当前帧顶点（唯一优化变量）
    // ------------------------------
    g2o::Sim3 cur_sim3(current_pose.Rotation, current_pose.Position, 1.0);
    VertexSim3* vertex_current = new VertexSim3();
    vertex_current->setId(n-1);
    vertex_current->setEstimate(cur_sim3);
    vertex_current->setFixed(false); // 设置为可优化
    optimizer.addVertex(vertex_current);
    k = 0;
    for (size_t i = n-2; i >= n-6; --i) {
        VertexSim3* vertex = new VertexSim3();
        vertex->setId(i);
        Eigen::Matrix3d Rlw = historic_poses[i].Rotation;
        Eigen::Vector3d tlw = historic_poses[i].Position;
        g2o::Sim3 Si(Rlw,tlw,1.0);

        vertex->setEstimate(Si);
        vertex->setFixed(true);
        Sij[k++] = Si;
        optimizer.addVertex(vertex);
    }
    // ------------------------------
    // 3. 添加前n帧的边（固定约束）
    // ------------------------------
    k = 0;
    for (size_t i = n-2; i >= n-6; --i) {
        // 创建边：连接当前帧顶点（id=0）和虚拟顶点（无实际顶点，仅作为约束）
        EdgeSim3* edge = new EdgeSim3();
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(n-1))); // 边仅关联当前帧顶点
        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i)));
        g2o::Sim3 Sli = Sij[k] * S_I_L * Sbibj[k] * S_I_L.inverse();
        // 设置测量值为相对位姿 Sim3
        edge->setMeasurement(Sij[k] * Sli.inverse());

        // 设置信息矩阵（根据传感器噪声调整，这里假设为单位矩阵）
        Eigen::Matrix<double, 7, 7> information = Eigen::Matrix<double, 7, 7>::Identity();
        edge->setInformation(information);

        optimizer.addEdge(edge);
        k++;
    }

    // ------------------------------
    // 4. 执行优化
    // ------------------------------
    optimizer.initializeOptimization();
    optimizer.optimize(10); // 迭代10次

    // ------------------------------
    // 5. 提取优化结果
    // ------------------------------
    g2o::Sim3 opted = vertex_current->estimate();
    current_pose.Rotation = opted.rotation().toRotationMatrix();
    current_pose.Position = opted.translation();

}
