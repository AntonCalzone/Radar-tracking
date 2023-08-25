import os
from src.tracking import RadarTracker


def by_index(file_name):
    """Return current frame index."""
    file_name = file_name.split('.')[0]
    return int(file_name[5:])


# Input
input_directory = './data'
input_files = os.listdir(input_directory)
input_files = [file for file in input_files if file.endswith('.csv')]
input_files.sort(key=by_index)
# TODO: Find good frames in dataset

tracker = RadarTracker()
i = 0

for file in input_files:
    tracker.get_detections_from_file(os.path.join(input_directory, file))
    tracker.iterate()
    tracker.visualize(raw_data_file=os.path.join(input_directory, file))
