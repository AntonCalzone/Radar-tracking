"""This module contains the RadarTracker class. RadarTracker is meant to iterate over sets of raw
radar detections, and keeps track of objects detected in the detections. It also classifies all
confirmed objects as moving or stationary."""
import pandas as pd
import numpy as np
from math import pi, cos, sin, atan, radians, degrees, acos
from .clustering2 import cluster_new_frame
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class RadarTracker:
    """Class for tracking objects based on radar detections. A "map" of all tracked object
    is maintained in a dataframe, with vessel in center and the y-axis aligned with true north."""

    def __init__(self, use_doppler=False, min_velocity=1):
        """Initializer for RadarTracker class.

        @param use_doppler  Enable/disable doppler, disabled by default
        @param min_velocity  Minimum velocity for object to be classified as moving, 1 m/s default.
        """
        # Declaration of constants
        self.distance_threshold = 30
        self.detection_threshold = 15
        self.score_threshold = 20
        self.score_increase = 5
        self.max_velocity = 40
        self.new_obj_position_tolerance = 50
        self.min_velocity = min_velocity

        # State variables
        self.detections = None  # Detections from radar
        # Currently tracked objects
        self.tracked_objects = pd.DataFrame(
            {},
            columns=["num_points", "xpos", "ypos", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y", "status", "id", "score", "max_score", "head", "vel", "est_xpos", "est_ypos", "est_top_left_x", "est_top_left_y", "est_bottom_right_x", "est_bottom_right_y", "last_update", "Classification"])
        self.current_heading = 0                   # Assume vessel pointed north when initializing
        self.doppler_enabled = use_doppler
        self.elapsed_time = None

        self.save_output = False
        self.num_frames_printed = 0

    def iterate(self, INS_data=(0, 0, 0), elapsed_time=1):
        """Perform tracking iteration.

        @param INS_data  Triple containing distance moved x-wise, y-wise, and rotation. (0,0,0)
        default
        @param elapsed_time  Time since last iteration, 1 by default"""
        self.detections = pd.DataFrame(self.detections)
        self.elapsed_time = elapsed_time
        self.current_heading = INS_data[2]

        # Update tracked objects based on vessel movement
        self.tracked_objects["last_update"] += self.elapsed_time
        self.rotate_detections()
        self.estimate_object_positions(INS_data)
        self.tracked_objects.sort_values("status")  # Sort on status to match confirmed objects first

        if False:   # More verbose and shorter way of mathing objects, but slower since it uses dataframes
            # Match old objects to new detections based on estimated position
            for i, old_object in enumerate(self.tracked_objects.itertuples()):
                for j, new_object in enumerate(self.detections.itertuples()):
                    if self.object_match(old_object, new_object):
                        self.update_object(i, new_object)
                        self.detections.drop(j, inplace=True)
                        self.detections.reset_index(inplace=True, drop=True)
                        break
        if True:
            if not self.tracked_objects.empty:

                # Assuming self.tracked_objects and self.detections are pandas DataFrames
                tracked_objects_np = self.tracked_objects.to_numpy().T
                detections_np = self.detections.to_numpy().T
                match_matrix_dim = (len(self.detections), len(self.tracked_objects))

                # Get position mask for tracked objects
                etlx_index = self.tracked_objects.columns.get_loc("est_top_left_x")
                ebrx_index = self.tracked_objects.columns.get_loc("est_bottom_right_x")
                etly_index = self.tracked_objects.columns.get_loc("est_top_left_y")
                ebry_index = self.tracked_objects.columns.get_loc("est_bottom_right_y")
                etlx_matrix = np.full(match_matrix_dim, tracked_objects_np[etlx_index])
                ebrx_matrix = np.full(match_matrix_dim, tracked_objects_np[ebrx_index])
                etly_matrix = np.full(match_matrix_dim, tracked_objects_np[etly_index])
                ebry_matrix = np.full(match_matrix_dim, tracked_objects_np[ebry_index])

                det_xpos = detections_np[self.detections.columns.get_loc("xpos")][:, np.newaxis]
                det_ypos = detections_np[self.detections.columns.get_loc("ypos")][:, np.newaxis]

                etlx_mask = etlx_matrix < det_xpos
                ebrx_mask = det_xpos < ebrx_matrix
                etly_mask = det_ypos < etly_matrix
                ebry_mask = ebry_matrix < det_ypos

                # Get position mask for new objects (objects w/ vel = -1)
                new_objects_indices = self.tracked_objects.columns.get_loc("vel")
                vel_matrix = np.full(match_matrix_dim, tracked_objects_np[new_objects_indices])
                new_objects = vel_matrix == -1
                etlx_mask_new = ((etlx_matrix - self.new_obj_position_tolerance / 2) < det_xpos) & new_objects
                ebrx_mask_new = (det_xpos < (ebrx_matrix + self.new_obj_position_tolerance / 2)) & new_objects
                etly_mask_new = (det_ypos < (etly_matrix + self.new_obj_position_tolerance / 2)) & new_objects
                ebry_mask_new = ((ebry_matrix - self.new_obj_position_tolerance / 2) < det_ypos) & new_objects

                # Total position mask
                position_mask = ((etlx_mask & ebrx_mask & etly_mask & ebry_mask) | (etlx_mask_new & ebrx_mask_new & etly_mask_new & ebry_mask_new)).T

                # Get size mask
                old_num_points = tracked_objects_np[self.tracked_objects.columns.get_loc("num_points")]
                new_num_points = detections_np[self.detections.columns.get_loc("num_points")]
                size_ratios = np.full(match_matrix_dim, old_num_points) / new_num_points[:, np.newaxis]

                size_lower_mask = size_ratios > 0.6
                size_upper_mask = size_ratios < 1.4
                size_mask = (size_lower_mask & size_upper_mask).T

                # Get match_matrix
                match_matrix = position_mask & size_mask

                # Find the first matching new object for each old object
                matched_indices = np.argmax(match_matrix, axis=1)
                matched_mask = match_matrix[np.arange(match_matrix.shape[0]), matched_indices]

                # Update matched objects
                updated_indices = []
                for i, index in enumerate(matched_indices):
                    if matched_mask[i] and index not in updated_indices:
                        self.update_object(i, self.detections.iloc[index])
                        updated_indices.append(index)

                # Remove matched detections and update the original DataFrame
                self.detections = self.detections.drop(index=matched_indices[matched_mask.astype(bool)]).reset_index(drop=True)

        self.tracked_objects["score"] -= self.score_increase

        # Update statuses, delete deprecated objects
        self.tracked_objects.loc[self.tracked_objects["score"] > self.detection_threshold, "status"] = "Confirmed"
        to_delete = np.where((self.tracked_objects["max_score"] - self.tracked_objects["score"] > self.score_increase) | (self.tracked_objects["score"] < 0))[0]
        self.tracked_objects.drop(to_delete, inplace=True)

        # Assign ID:s for objects confirmed this iteration
        num_tracks_to_id = len(self.tracked_objects.loc[(self.tracked_objects["id"] == -1) & (self.tracked_objects["status"] == "Confirmed")])
        start_id = max(self.tracked_objects["id"].to_list() + [0])
        self.tracked_objects.loc[(self.tracked_objects["id"] == -1) & (self.tracked_objects["status"] == "Confirmed"), "id"] = [i for i in range(start_id, start_id + num_tracks_to_id)]

        # Non-matched new detections are added as tentative objects
        missing_columns = ["status", "id", "score", "max_score", "est_xpos", "est_ypos", "vel", "vel0", "vel1", "vel2", "vel3", "vel4", "head", "head0", "head1", "head2", "head3", "head4", "last_update"]
        missing_values = ["Tentative", -1, 0, 0, 0, 0, -1, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan, np.nan, 0]
        for col, val in zip(missing_columns, missing_values):
            self.detections[col] = val
        self.tracked_objects = pd.concat([self.tracked_objects, self.detections], ignore_index=True, sort=False)

        # Classify objects
        self.classify()

    def classify(self):
        """Classify all tracked objects as undefined (0), stationary(1), or moving (2)"""
        self.tracked_objects["Classification"] = 0
        self.tracked_objects.loc[(self.tracked_objects["status"] == "Confirmed") & (self.tracked_objects["vel"] < 1), "Classification"] = 1
        self.tracked_objects.loc[(self.tracked_objects["status"] == "Confirmed") & (self.tracked_objects["vel"] > 1), "Classification"] = 2

        for index, tracked_object in enumerate(self.tracked_objects.itertuples()):
            if inconsistent_heading(tracked_object):
                self.tracked_objects.at[index, 'Classification'] = 1

    def estimate_object_positions(self, INS_data):
        """Update tracked objects estimated positions.

        Position estimation is done via dead reckoning, if no recorded velocity/heading the
        estimated position is set to detected position. Takes vessel movement into account.

        @param INS_data  Triple describing vessels movement since last iteration"""
        # Set all estimated values to true values
        self.tracked_objects["est_xpos"] = self.tracked_objects["xpos"]
        self.tracked_objects["est_ypos"] = self.tracked_objects["ypos"]
        self.tracked_objects["est_top_left_y"] = self.tracked_objects["top_left_y"]
        self.tracked_objects["est_bottom_right_y"] = self.tracked_objects["bottom_right_y"]
        self.tracked_objects["est_top_left_x"] = self.tracked_objects["top_left_x"]
        self.tracked_objects["est_bottom_right_x"] = self.tracked_objects["bottom_right_x"]

        # Set estimated values of objects w/ recorded velocities by dead reckoning
        self.tracked_objects.loc[self.tracked_objects["vel"] != -1, "est_xpos"] = \
            self.tracked_objects["xpos"] + self.tracked_objects["vel"] * np.cos(self.tracked_objects["head"].astype(float)) * self.elapsed_time
        self.tracked_objects.loc[self.tracked_objects["vel"] != -1, "est_ypos"] = \
            self.tracked_objects["ypos"] + self.tracked_objects["vel"] * np.sin(self.tracked_objects["head"].astype(float)) * self.elapsed_time
        self.tracked_objects.loc[self.tracked_objects["vel"] != -1, "est_top_left_y"] = \
            self.tracked_objects["top_left_y"] + self.tracked_objects["vel"] * np.sin(self.tracked_objects["head"].astype(float)) * self.elapsed_time
        self.tracked_objects.loc[self.tracked_objects["vel"] != -1, "est_bottom_right_y"] = \
            self.tracked_objects["bottom_right_y"] + self.tracked_objects["vel"] * np.sin(self.tracked_objects["head"].astype(float)) * self.elapsed_time
        self.tracked_objects.loc[self.tracked_objects["vel"] != -1, "est_top_left_x"] = \
            self.tracked_objects["top_left_x"] + self.tracked_objects["vel"] * np.cos(self.tracked_objects["head"].astype(float)) * self.elapsed_time
        self.tracked_objects.loc[self.tracked_objects["vel"] != -1, "est_bottom_right_x"] = \
            self.tracked_objects["bottom_right_x"] + self.tracked_objects["vel"] * np.cos(self.tracked_objects["head"].astype(float)) * self.elapsed_time

        # Account for distance travelled in true and estimated position
        x_diff, y_diff, _ = INS_data
        self.tracked_objects[["xpos", "est_xpos"]] -= x_diff
        self.tracked_objects[["ypos", "est_ypos"]] -= y_diff

    def rotate_detections(self):
        """Rotate newly detected objects to align true north and y-axis."""
        theta = radians(- self.current_heading)
        rotated_detections = self.detections.copy(deep=True)

        # Rotate center points
        rotated_detections['xpos'] = self.detections['xpos'] * cos(theta) - self.detections['ypos'] * sin(theta)
        rotated_detections['ypos'] = self.detections['ypos'] * cos(theta) + self.detections['xpos'] * sin(theta)
        # Rotate all corner points
        rotated_detections['top_left_y'] = self.detections['top_left_y'] * cos(theta) + self.detections['top_left_x'] * sin(theta)
        rotated_detections['bottom_left_y'] = self.detections['bottom_left_y'] * cos(theta) + self.detections['bottom_left_x'] * sin(theta)
        rotated_detections['top_right_y'] = self.detections['top_right_y'] * cos(theta) + self.detections['top_right_x'] * sin(theta)
        rotated_detections['bottom_right_y'] = self.detections['bottom_right_y'] * cos(theta) + self.detections['bottom_right_x'] * sin(theta)
        rotated_detections['top_left_x'] = self.detections['top_left_x'] * cos(theta) - self.detections['top_left_y'] * sin(theta)
        rotated_detections['bottom_left_x'] = self.detections['bottom_left_x'] * cos(theta) - self.detections['bottom_left_y'] * sin(theta)
        rotated_detections['top_right_x'] = self.detections['top_right_x'] * cos(theta) - self.detections['top_right_y'] * sin(theta)
        rotated_detections['bottom_right_x'] = self.detections['bottom_right_x'] * cos(theta) - self.detections['bottom_right_y'] * sin(theta)
        # Save new top_right/bottom_left values, drop other values
        rotated_detections['top_left_y'] = rotated_detections[['top_left_y', 'bottom_left_y', 'top_right_y', 'bottom_right_y']].max(axis=1)
        rotated_detections['bottom_right_y'] = rotated_detections[['top_left_y', 'bottom_left_y', 'top_right_y', 'bottom_right_y']].min(axis=1)
        rotated_detections['top_left_x'] = rotated_detections[['top_left_x', 'bottom_left_x', 'top_right_x', 'bottom_right_x']].min(axis=1)
        rotated_detections['bottom_right_x'] = rotated_detections[['top_left_x', 'bottom_left_x', 'top_right_x', 'bottom_right_x']].max(axis=1)
        rotated_detections.drop(['bottom_left_x', 'bottom_left_y', 'top_right_x', 'top_right_y'], axis=1)

        self.detections = rotated_detections

    def object_match(self, old_object, new_object):
        """Check if two objects are similair enough to be matched.

        @param old_object  Previously tracked object
        @param new_object  Newly detected object"""
        # Size check, true if objects within 30% of eachother. NOTE: Might have to tweak range
        size_ratio = old_object.num_points / new_object.num_points
        if not (0.7 < size_ratio < 1.3):
            return False

        # Distance check, true if new object within old objects estimated bounding box
        if not (old_object.est_top_left_x < new_object.xpos < old_object.est_bottom_right_x) or not (old_object.est_bottom_right_y < new_object.ypos < old_object.est_top_left_y):
            return False

        # 0=not moving, 1=receding, 2=approaching, 255=doppler not on
        # Doppler check, made when doppler enabled, new object has doppler value, and old object has velocity/heading
        # NOTE: Might have to tweak ranges for receding/approaching objects after testing
        # TODO: Use a minimum value for speed? Maybe slow objects don't register on doppler
        if self.doppler_enabled and (new_object.doppler != 255) and (old_object.vel != -1):
            if old_object.vel < self.min_velocity:  # If object stationary, check that doppler confirms
                return (new_object.doppler == 0)
            else:
                # Calculate angle between vector from origin to new_object and heading vector of old_object using dot product
                origin_vector = [new_object.xpos, new_object.ypos]
                heading_vector = [new_object.xpos * cos(old_object.head), new_object.ypos * sin(old_object.head)]
                angle = degrees(acos(np.dot(origin_vector, heading_vector) / (np.linalg.norm(origin_vector) * np.linalg.norm(heading_vector))))
                if angle < 75:
                    return (new_object.doppler == 2)     # If angle < 75deg object should be approaching
                elif angle > 105:
                    return (new_object.doppler == 1)     # If angle > 105deg object should be receding
                else:
                    return (new_object.doppler == 0)     # If 75deg < angle < 105deg object should not be moving
        else:
            return True

    def get_velocity_heading(self, old_object, new_object):
        """Calculate velocity and heading of object based on new detection.

        Returns average of last five velocities/heading to reduce noise.

        @param old_object  Previously tracked object
        @param new_object  Newly detected object"""
        # Calculate current velocity
        time_since_update = old_object.last_update
        distance_travelled = ((old_object.xpos - new_object.xpos)**2 + (old_object.ypos - new_object.ypos)**2)**0.5
        current_vel = distance_travelled / time_since_update

        # Calculate current heading
        if old_object.xpos == new_object.xpos:
            if old_object.ypos >= new_object.ypos:    # Avoid dividing w/ zero
                current_head = - pi / 2
            else:
                current_head = pi / 2
        else:
            opposite = new_object.ypos - old_object.ypos
            adjacent = new_object.xpos - old_object.xpos
            current_head = atan(opposite / adjacent)
            if opposite < 0 and adjacent < 0:   # If in 3d quadrant
                current_head -= pi
            elif adjacent < 0:                  # If in 4th quadrant
                current_head -= pi

        # Get five last recorded values of velocity/heading
        vel_list = [current_vel]
        head_list = [current_head]
        for i in range(4):
            vel_list.append(old_object[self.tracked_objects.columns.get_loc("vel" + str(i))])
            head_list.append(old_object[self.tracked_objects.columns.get_loc("head" + str(i))])

        vel_list = np.array(vel_list)
        head_list = np.array(head_list)
        vel = np.nanmean(vel_list)
        head = np.nanmean(head_list)

        return vel, vel_list, head, head_list

    def update_object(self, old_object_index, new_object):
        """Update tracked object based on new spotting.

        @param old_object_index  Index in dataframe of object
        @param new_object  Newly spotted object"""
        # Calculate new values
        num_points = new_object.num_points
        est_xpos = self.tracked_objects.loc[old_object_index, "xpos"]
        est_ypos = self.tracked_objects.loc[old_object_index, "ypos"]
        score = self.tracked_objects.loc[old_object_index, "score"] + 2 * self.score_increase
        vel, vel_history, head, head_history = self.get_velocity_heading(self.tracked_objects.loc[old_object_index], new_object)
        max_score = max(self.tracked_objects.loc[old_object_index, "max_score"], score - self.score_increase)

        # Assign new values to old object
        columns_to_update = ("num_points", "xpos", "ypos", "top_left_x", "bottom_right_x", "bottom_right_y", "top_left_y", "est_xpos", "est_ypos", "score", "vel", "vel0", "vel1", "vel2", "vel3", "vel4", "head", "head0", "head1", "head2", "head3", "head4", "last_update", "max_score")
        new_values = [num_points, new_object.xpos, new_object.ypos, new_object.top_left_x, new_object.bottom_right_x, new_object.bottom_right_y, new_object.top_left_y, est_xpos, est_ypos, score, vel, *vel_history, head, *head_history, 0, max_score]
        self.tracked_objects.loc[old_object_index, columns_to_update] = new_values

    def enable_doppler(self):
        """Enable doppler functionality."""
        self.doppler_enabled = True

    def disable_doppler(self):
        """Disable doppler functionality."""
        self.doppler_enabled = False

    def set_min_velocoty(self, min_velocity):
        """Set minimum velocity for object to be considered moving.

        @param min_velocity  New minimum velocity"""
        self.min_velocity = min_velocity

    def get_detections_from_dataframe(self, detections_df):
        """Update radar detections with dataframe.

        @param detections_df  Dataframe containing new radar detections"""
        self.detections = cluster_new_frame(detections_df)

    def get_detections_from_numpy_dict(self, detections_dict):
        """Update radar detections with numpy dictionary

        @param detections_dict  Dictionary containing new radar detections"""
        self.detections = cluster_new_frame(detections_dict)

    def get_detections_from_file(self, file_name):
        """Update radar detections with data from csv file.

        @param file_name  Path to file containing new radar detections"""
        # Check if file contains doppler data or not
        with open(file_name, 'r') as f:
            doppler_included = len(f.readlines()[0].split(',')) == 4

        if doppler_included:
            detections_df = pd.read_csv(file_name, names=['x', 'y', 'intensity', 'doppler'])
        else:
            detections_df = pd.read_csv(file_name, names=['x', 'y', 'intensity'])
            detections_df['doppler'] = 255

        np_lists = detections_df.to_numpy().T   # Convert df to np lists
        detections_dict = {'x': np_lists[0], 'y': np_lists[1], 'intensity': np_lists[2], 'doppler': np_lists[3]}
        self.get_detections_from_numpy_dict(detections_dict)

    def get_tracked_objects(self):
        """Return iterator of currently tracked objects.

        Used when publishing confirmed objects in ROS."""
        return self.tracked_objects.itertuples()

    def get_number_of_tracked_objects(self):
        """Return number of tracked objects."""
        return len(self.tracked_objects)

    def get_number_of_confirmed_objects(self):
        """Return number of confirmed objects."""
        return len(self.tracked_objects.loc[self.tracked_objects["status"] == "Confirmed"])

    def visualize(self, raw_data_file=None):
        """Visualize currently tracked objects in map.

        @param raw_data_file  Path to file containing raw radar detections, if not given only
        tracked objects will be plotted"""
        arrow_length = 30

        # Clear any open figures
        plt.clf()
        plt.cla()

        # Scatter plot raw radar detections if file name supplied
        if raw_data_file is not None:
            points = pd.read_csv(raw_data_file, index_col=False, names=['x', 'y', 'intensity', 'doppler'])

            # Rotate data to align y-axis and true north
            theta = radians(- self.current_heading)
            rotated_points = points.copy(deep=True)
            rotated_points['x'] = points['x'] * cos(theta) - points['y'] * sin(theta)
            rotated_points['y'] = points['y'] * cos(theta) + points['x'] * sin(theta)

            plt.scatter(rotated_points['x'], rotated_points['y'], s=0.1)

        # Set color of objects
        self.tracked_objects["color"] = 'r'
        self.tracked_objects.loc[self.tracked_objects["Classification"] == 1, "color"] = '#f2f500'
        self.tracked_objects.loc[self.tracked_objects["Classification"] == 2, "color"] = '#00ff00'
        self.tracked_objects["point_size"] = self.tracked_objects["num_points"] / 20

        # Draw MARV heading arrow
        x, y = 0, 0
        dx = 15 * cos(radians(90 - self.current_heading))
        dy = 15 * sin(radians(90 - self.current_heading))
        plt.arrow(x, y, dx, dy, length_includes_head=True)
        plt.scatter([0], [0], s=5, c='r')    # Draw red points for MARV

        plt.scatter(self.tracked_objects['xpos'], self.tracked_objects['ypos'], c=self.tracked_objects["color"])
        ax = plt.gca()
        for row in self.tracked_objects.itertuples():
            if row.status == "Confirmed":                         # Draw bounding boxes
                width = row.bottom_right_x - row.top_left_x
                height = row.top_left_y - row.bottom_right_y
                x = row.top_left_x
                y = row.bottom_right_y
                ax.add_patch(Rectangle((x, y), width, height, edgecolor='k', fill=False))
                if row.Classification == 2:     # Draw heading arrows for moving objects
                    x = row.xpos
                    y = row.ypos
                    dx = arrow_length * np.cos(row.head)
                    dy = arrow_length * np.sin(row.head)
                    text_dx = (arrow_length + 5) * np.cos(row.head)
                    text_dy = (arrow_length + 5) * np.sin(row.head)
                    plt.arrow(x, y, dx, dy, length_includes_head=True)
                    plt.text(x + text_dx, y + text_dy, str(round(row.vel, 2)), fontsize=5)
                    plt.text(x, y, str(row.id), fontsize=6)
        plt.xlim([-450, 450])
        plt.ylim([-450, 450])

        if self.save_output:
            file_name = "/Users/anton/Desktop/real_run/frame" + str(self.num_frames_printed) + ".png"
            plt.savefig(file_name)
            self.num_frames_printed += 1
        plt.show()
        # plt.waitforbuttonpress()
        # TODO: plt.waitforbuttonpress???


# Other Functions
def inconsistent_heading(object):
    """Check if heading of object is inconsistent.

    @param object  Object to check heading of"""
    if object.status == "Confirmed":
        if abs(object.head4 - object.head3) > pi or abs(object.head3 - object.head2) > pi or abs(object.head2 - object.head1) > pi or abs(object.head1 - object.head0) > pi:
            return True
    return False
