import numpy as np
from scipy.optimize import least_squares


class OutOfSensorBoundsError(Exception):
    pass


class Camera(object):

    def __init__(self, f_length, sensor_size):

        self.p = None  # Pose
        self.p0 = None
        self.f = f_length  # Focal Length in Pixels
        self.sensor_size = sensor_size

        # Throw error if projected coord is out of sensor bounds
        self.error_on_oob = False

    def projective_transform(self, x):
        """
        This function performs the projective transform on generalized coordinates in the camera reference frame.
        """

        x = np.asarray(x)
        # Assume no intensity column
        x0, y0, z0 = x

        # Camera coors to pixel coors
        u = ((x0 / z0) * self.f) + (self.sensor_size[0] // 2)
        v = ((y0 / z0) * self.f) + (self.sensor_size[1] // 2)

        u_min = np.min(u)
        v_min = np.min(v)

        n = len(u)
        u_list = []
        v_list = []
        if self.error_on_oob:
            for i in range(n):
                if (u[i] >= u_min and u[i] <= self.sensor_size[0] and v[i] >= v_min and v[i] <= self.sensor_size[1]):
                    u_list.append(u[i])
                    v_list.append(v[i])
                else:
                    raise OutOfSensorBoundsError("Projected coordinate was outside the sensor")
        else:
            for i in range(n):
                u_list.append(u[i])
                v_list.append(v[i])

        u = np.asarray(u_list)
        v = np.asarray(v_list)

        return np.vstack((u, v))

    @staticmethod
    def make_cam_mtx(fi, theta, psi, translation_vec):

        translation_mtx = np.array([[1, 0, 0, -translation_vec[0]],
                                    [0, 1, 0, -translation_vec[1]],
                                    [0, 0, 1, -translation_vec[2]],
                                    [0, 0, 0, 1]])

        # Apply yaw. It represents rotation around camera's z axis
        cos_fi = np.cos(fi)
        sin_fi = np.sin(fi)
        R_yaw = np.array([[cos_fi, -sin_fi, 0, 0],
                          [sin_fi, cos_fi, 0, 0],
                          [0, 0, 1, 0]])

        # Apply pitch. Represents rotation around camera's x axis
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_pitch = np.array([[1, 0, 0],
                            [0, cos_theta, sin_theta],
                            [0, -sin_theta, cos_theta]])

        # Apply roll. Represents rotation around camera's y axis
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        R_roll = np.array([[cos_psi, 0, -sin_psi],
                           [0, 1, 0],
                           [sin_psi, 0, cos_psi]])

        R_ax_swp = np.array([[1, 0, 0],
                             [0, 0, -1],
                             [0, 1, 0]])

        return R_ax_swp @ R_roll @ R_pitch @ R_yaw @ translation_mtx

    def rotational_transform(self, X):
        """
        This function performs the translation and rotation from world coordinates into generalized camera coordinates.
        """

        # Unpack pose? could do something different here.
        X_cam, Y_cam, Z_cam, azimuth_cam_deg, pitch_cam_deg, roll_cam_deg = self.p

        # Make X a set of homogeneous coors
        X = np.vstack((X, np.ones(X.shape[1])))

        # Convert degrees to radians
        azimuth_cam_rad = np.deg2rad(azimuth_cam_deg)
        pitch_cam_rad = np.deg2rad(pitch_cam_deg)
        roll_cam_rad = np.deg2rad(roll_cam_deg)

        translation_vec = [X_cam, Y_cam, Z_cam]
        C = self.make_cam_mtx(azimuth_cam_rad, pitch_cam_rad, roll_cam_rad, translation_vec)

        return C @ X

    def estimate_pose(self, X_gcp, u_gcp, p0):
        """
        This function adjusts the pose vector such that the difference between the observed pixel coordinates u_gcp
        and the projected pixels coordinates of X_gcp is minimized.
        """

        self.p = p0.copy()

        def residuals(p, X_gcp, u_gcp):
            self.p = p.copy()
            xuv = self.ene_to_camera(X_gcp)
            return xuv.ravel() - u_gcp.ravel()

        res = least_squares(residuals, self.p, args=(X_gcp, u_gcp))
        self.p = res.x

    # f
    def ene_to_camera(self, X):
        return self.projective_transform(self.rotational_transform(X))


def read_gcp(fname):
    u = []
    v = []
    east = []
    north = []
    ele = []
    with open(fname) as fd:
        for i, line in enumerate(fd):
            # if i < 2 or line.startswith("#"):
            #     skip header and commented lines
                # continue
            vals = line.split(",")[1:-1]
            u.append(int(float(vals[0].strip())))
            v.append(int(float(vals[1].strip())))
            east.append(float(vals[2].strip()))
            north.append(float(vals[3].strip()))
            ele.append(float(vals[4].strip()))
    uv = np.vstack((u, v))
    ene = np.vstack((east, north, ele))
    return uv, ene


def read_gcp_nh(gcp_fname):

    uv_ene = np.loadtxt(gcp_fname, delimiter=',')
    uv, ene = uv_ene[:, 0:2], uv_ene[:, 2:]

    return uv, ene
