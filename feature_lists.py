# shock onset times
onset_times = ['id_key', 'dbp_0', 'hr_0', 'rr_0', 'sbp_0', 'spo2_0', 'tempc_0']

# list of vital signs
feature_names = ['dbp_', 'hr_', 'rr_', 'sbp_', 'spo2_', 'tempc_', 'map_', 'si_']

# data hours available
hour_list = [1, 2, 3, 4, 5, 6]


def get_onset_times():
    return onset_times.copy()


def get_feature_names():
    return feature_names.copy()


def get_hour_list():
    return hour_list.copy()

