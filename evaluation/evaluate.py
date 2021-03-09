import pandas as pd
import numpy as np
from psds_eval import PSDSEval

sample_length = 10.0

def check_format(row, labels, known_labels):
    # NaN check
    if labels.isnull().values.any():
        return False, "Some values are missing or are NaN in row {}".format(row)

    # start must not be before zero
    if labels.loc["onset"] < 0:
        return False, "Row {}: Starts before zero".format(row)

    # end must not be too far off
    if labels.loc["offset"] > sample_length:
        return False, "Row {}: End beyond the end of the data".format(row)

    # end must be larger than start
    if labels['onset'] > labels['offset']:
        return False, "Start of row {} must be smaller than end".format(row)

    # only known labels
    if not labels["event_label"] in known_labels:
        return False, "Row {} contains unknown labels".format(row)

    return True, "OK"


def _load_trials(file_path):
    # read file
    label_data = pd.read_csv(file_path, header=0, usecols=[0,1,2,3])

    # known labels
    label_set = set(label_data["event_label"].unique())

    # known files
    trial_set = set(label_data["filename"].unique())

    return label_data, label_set, trial_set


def score_all(hyp_data, ref_data):
    # as the unique files are the same, only one unique call is enough
    metadata = pd.DataFrame(np.unique(hyp_data['filename']), columns=['filename'])
    metadata = metadata.assign(duration=10)

    psds_eval = PSDSEval(ground_truth=ref_data, metadata=metadata)
    info = {"threshold": 0.5}
    psds_eval.add_operating_point(hyp_data, info=info)
    return psds_eval.psds(max_efpr=100).value


def evaluate(prediction_file, ground_truth_file):
    ref_data, ref_label_set, ref_trial_set = _load_trials(ground_truth_file)
    try:
        hyp_data, hyp_label_set, hyp_trial_set = _load_trials(prediction_file)
        # files check
        if ref_trial_set != hyp_trial_set:
            difference = ref_trial_set - hyp_trial_set
            return float('nan'), "Each file needs to have at least one label. {} file(s) missing, e.g. {}".format(len(difference), list(difference)[:3])
        # dtypes check
        if not np.all(hyp_data.dtypes == np.array([np.dtype('O'), np.dtype('float64'), np.dtype('float64'), np.dtype('O')])):
            return float('nan'), "Incorrect data types. Make sure the csv order is correct and each column uses the correct type."

        for index, row in hyp_data.iterrows():
            check_ok, check_msg = check_format(index, row, ref_label_set)
            if not check_ok:
                return float('nan'), check_msg
    except Exception as e:
        return float('nan'), str(e)

    try:
        score = score_all(hyp_data, ref_data)
    except Exception as err:
        print(err)
        return float('nan'), "Could not calculate error."

    return score, 'Valid file'
