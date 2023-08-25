from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter
import time

epsilon = 5 
min_points = 15 


def cluster_new_frame(detections):
    """Return dictionary containing data about detected clusters in radar detections.
    
    Clustering is done with DBSCAN algorithm.
    @param detections  Dictionary containing raw radar data"""

    # start_time = time.perf_counter()
    clustering = DBSCAN(eps=epsilon,min_samples=min_points,n_jobs=-1).fit(np.array([detections['x'],detections['y']]).T)
    # cluster_time = time.perf_counter()

    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)   #Get unique labels and count of labels
    object_indices = [(labels == label).nonzero()[0] for label in unique_labels]   # Get indices of detections belonging to each object
    num_objects = len(unique_labels)

    if (-1) in unique_labels:       #Remove noise points if present
        counts = counts[1:]
        num_objects -= 1
        object_indices = object_indices[1:]


    # Initiatie output dict (filled to save time)
    output = {"num_points":np.empty(num_objects,dtype=np.float16),"xpos":np.empty(num_objects,dtype=np.float16),
              "ypos":np.empty(num_objects,dtype=np.float16),"doppler":np.empty(num_objects,dtype=np.float16),
              "top_left_x":np.empty(num_objects,dtype=np.float16),"top_left_y":np.empty(num_objects,dtype=np.float16),
              "bottom_left_x":np.empty(num_objects,dtype=np.float16),"bottom_left_y":np.empty(num_objects,dtype=np.float16),
              "top_right_x":np.empty(num_objects,dtype=np.float16),"top_right_y":np.empty(num_objects,dtype=np.float16),
              "bottom_right_x":np.empty(num_objects,dtype=np.float16),"bottom_right_y":np.empty(num_objects,dtype=np.float16)}

    # Fill output dict
    output["num_points"] = counts
    for i, (indices, count) in enumerate(zip(object_indices, counts)):
        output["xpos"][i] = np.sum(detections['x'][indices]) / count 
        output["ypos"][i] = np.sum(detections['y'][indices]) / count
        output["top_left_x"][i] = np.amin(detections['x'][indices])
        output["top_left_y"][i] = np.amax(detections['y'][indices])
        output["bottom_left_x"][i] = np.amin(detections['x'][indices])
        output["bottom_left_y"][i] = np.amin(detections['y'][indices])
        output["top_right_x"][i] = np.amax(detections['x'][indices])
        output["top_right_y"][i] = np.amax(detections['y'][indices])
        output["bottom_right_x"][i] = np.amax(detections['x'][indices])
        output["bottom_right_y"][i] = np.amin(detections['y'][indices])

        doppler_values, doppler_counts = np.unique(detections['doppler'], return_counts=True)
        output["doppler"][i] = doppler_values[np.argmax(doppler_counts)]    

    # end_time = time.perf_counter()       
    return output
    # return (output, cluster_time - start_time, end_time - cluster_time)

def get_majority(elements):
    """Returns majority value and count of value given list of elements"""
    counter = Counter(elements)
    value, count = counter.most_common()[0]
    return value, count
