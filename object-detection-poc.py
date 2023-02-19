import numpy as np
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter

def deg_to_rad(deg):
    return deg * 0.0174533

#this function converts coordinates
def convert_coords(obj_polar_coords, target_horiz_dist = 1, target_vertical_dist = 5):
    angle = np.arctan(((np.cos(deg_to_rad(obj_polar_coords[1])) * obj_polar_coords[0]) - target_horiz_dist)/((np.sin(deg_to_rad(obj_polar_coords[1])) * obj_polar_coords[0]) + target_vertical_dist))
    dist = ((np.cos(deg_to_rad(obj_polar_coords[1])) * obj_polar_coords[0]) - target_horiz_dist) / np.cos(angle)
    return (dist, angle)

def main():
    #loading data
    lidar_data = np.loadtxt("test_lidar_data.txt");
    
    #converting polar coords to euclidean coords
    X_data = np.zeros((len(lidar_data), 2))
    for i in range(len(lidar_data)):
        X_data[i, 0] = lidar_data[i, 0] * np.cos(lidar_data[i, 1])
        X_data[i, 1] = lidar_data[i, 0] * np.sin(lidar_data[i, 1])

    #cluster params
    eps = 0.5
    min_samples = 5

    #clustering data
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbs.fit_predict(X_data)

    print(labels)

    kalman_filters = []
    unique_labels = np.unique(labels, return_counts=True)

    #loop over objects
    for l in unique_labels[0]:
        #skip noise
        if l == -1:
            continue

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([0, 0, 0, 0])  # initial state (x, y, vx, vy)
        kf.F = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])  # state transition matrix
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])  # measurement function
        kf.P = np.diag([1, 1, 10, 10])  # initial covariance
        kf.R = np.diag([0.1, 0.1])  # measurement noise
        kf.Q = np.diag([0.01, 0.01, 0.1, 0.1])  # process noise

        kalman_filters.append(kf)
    

if __name__ == "__main__":
    main()