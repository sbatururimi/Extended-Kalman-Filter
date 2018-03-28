#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {

    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    /**
     TODO:
     * update the state by using Kalman Filter equations
     */    
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;
    
    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    // To calculate  y, we use the equations that map the predicted location x′
    //  from Cartesian coordinates to polar coordinates
    float px = x_[0];
    float py = x_[1];
    float vx = x_[2];
    float vy = x_[3];
    
    float p1 = sqrt(px * px + py * py);
    
    float ro = p1;
    float phi = atan2(py, px);
    float ro_dot = (px * vx + py * vy) / p1;
    VectorXd h_(3);
    h_ << ro, phi, ro_dot;
    
    
    VectorXd y = z - h_;
    
    // Normalizing Angles
    phi = y[1];
    while (phi < -M_PI || phi > M_PI) {
        if (phi < -M_PI) {
            phi = phi +  2 * M_PI;
        }
        if (phi > M_PI) {
            phi = phi - 2 * M_PI;
        }
    }
    y[1] = phi;

    MatrixXd Ht = H_.transpose(); // H_ is the Jacobian matrix here
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;
    
    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
