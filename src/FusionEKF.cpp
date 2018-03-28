#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;
    
    previous_timestamp_ = 0;
    
    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);
    
    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;
    
    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;
    
    //state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1000, 0,
    0, 0, 0, 1000;
    
    //measurement matrix
    H_laser_ = MatrixXd(2, 4);
    H_laser_ <<
    1, 0, 0, 0,
    0, 1, 0, 0;
    
    //the initial transition matrix F_
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ <<
    1, 0, 1, 0,
    0, 1, 0, 1,
    0, 0, 1, 0,
    0, 0, 0, 1;
    
    noise_ax_ = 9;
    noise_ay_ = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {

        // first measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;
        
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
             Convert radar from polar to cartesian coordinates and initialize state.
             */
            float ro = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            if(phi > M_PI || phi < -M_PI){
                cout << "Phi out of range" << endl;
            }
            float px = ro * cos(phi);
            float py = ro * sin(phi);
            float vx = 0;
            float vy = 0;
            
            ekf_.x_ << px, py, vx, vy;
            
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
             Initialize state.
             */
            //set the state with the initial location and zero velocity
            float px = measurement_pack.raw_measurements_[0];
            float py = measurement_pack.raw_measurements_[1];
            float vx = 0;
            float vy = 0;
            
            ekf_.x_ << px, py, vx, vy;

        }
        
        // done initializing, no need to predict or update
        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        
        return;
    }
    
    /*****************************************************************************
     *  Prediction
     ****************************************************************************/
    
    
    //compute the time elapsed between the current and previous measurements
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;    //dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;
    
    // update the state transition matrix
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;
    
    //set the process covariance matrix Q
    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;
    
    
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ <<
    dt_4 / 4 * noise_ax_, 0, dt_3 / 2 * noise_ax_, 0,
    0, dt_4 / 4 * noise_ay_, 0, dt_3 / 2 * noise_ay_,
    dt_3 / 2 * noise_ax_, 0, dt_2 * noise_ax_, 0,
    0, dt_3 / 2 * noise_ay_, 0, dt_2 * noise_ay_;

    ekf_.Predict();
    
    
    /*****************************************************************************
     *  Update
     ****************************************************************************/

    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        Hj_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.H_ = Hj_;
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        // Laser updates
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }
    
    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
