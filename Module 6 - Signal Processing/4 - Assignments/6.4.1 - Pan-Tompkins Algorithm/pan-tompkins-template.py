import numpy as np
from pathlib import Path
from ekg_testbench import EKGTestBench
from scipy.signal import butter, find_peaks, sosfiltfilt

def detect_heartbeats(filepath):
    """
    Perform analysis to detect location of heartbeats
    :param filepath: A valid path to a CSV file of heart beats
    :return: signal: a signal that will be plotted
    beats: the indices of detected heartbeats
    """
    if filepath == '':
        return np.array([]), np.array([], dtype=int)

    # load data in matrix from CSV file; skip first two rows
    data = np.loadtxt(filepath, delimiter=',', skiprows=2)

    # save each vector as own variable
    time = data[:, 0]
    sample_rate = 1 / np.median(np.diff(time))

    # identify one column to process. Call that column signal
    signal = data[:, 1]

    # pass data through LOW PASS FILTER (OPTIONAL)
    # A band-pass filter combines the low-pass and high-pass stages.
    upper_cutoff = min(15.0, sample_rate * 0.45)
    filter_coefficients = butter(
        3,
        [5.0, upper_cutoff],
        btype='bandpass',
        fs=sample_rate,
        output='sos',
    )
    filtered_signal = sosfiltfilt(filter_coefficients, signal)

    # pass data through differentiator
    differentiated_signal = np.diff(filtered_signal, prepend=filtered_signal[0]) * sample_rate

    # pass data through square function
    squared_signal = differentiated_signal ** 2

    # pass through moving average window
    window_size = max(1, round(0.15 * sample_rate))
    integration_window = np.ones(window_size) / window_size
    signal = np.convolve(squared_signal, integration_window, mode='same')

    # use find_peaks to identify peaks within averaged/filtered data
    # save the peaks result and return as part of testbench result

    detection_threshold = np.quantile(signal, 0.8)
    beats, _ = find_peaks(
        signal,
        height=detection_threshold,
        distance=max(1, round(0.25 * sample_rate)),
        prominence=0.25 * detection_threshold,
    )

    # do not modify this line
    return signal, beats


# when running this file directly, this will execute first
if __name__ == "__main__":

    # place here so doesn't cause import error
    import matplotlib.pyplot as plt

    # database name
    database_name = 'mitdb_201'

    # set to true if you wish to generate a debug file
    file_debug = False

    # set to true if you wish to print overall stats to the screen
    print_debug = True

    # set to true if you wish to show a plot of each detection process
    show_plot = False

    ### DO NOT MODIFY BELOW THIS LINE!!! ###

    # path to ekg folder
    path_to_folder = Path(__file__).resolve().parents[3] / "data" / "ekg"

    # select a signal file to run
    signal_filepath = path_to_folder / (database_name + ".csv")

    # call main() and run against the file. Should return the filtered
    # signal and identified peaks
    (signal, peaks) = detect_heartbeats(signal_filepath)

    # stop with a clear message if detect_heartbeats() is not finished yet
    if peaks is None:
        raise SystemExit("detect_heartbeats() returned None for beats. Finish the steps above and return the peak indices.")

    # matched is a list of (peak, annotation) pairs; unmatched is a list of peaks that were
    # not matched to any annotation; and remaining is annotations that were not matched.
    annotation_path = path_to_folder / (database_name + "_annotations.txt")
    tb = EKGTestBench(annotation_path)
    peaks_list = peaks.tolist()
    (matched, unmatched, remaining) = tb.generate_stats(peaks_list)

    # if was matched, then is true positive
    true_positive = len(matched)

    # if response was unmatched, then is false positive
    false_positive = len(unmatched)

    # whatever remains in annotations is a missed detection
    false_negative = len(remaining)

    # calculate f1 score
    f1 = true_positive / (true_positive + 0.5 * (false_positive + false_negative))

    # if we wish to show the resulting plot
    if show_plot:
        # make a nice plt of results
        plt.title('Signal for ' + database_name + " with detections")

        plt.plot(signal, label="Filtered Signal")
        plt.plot(peaks, signal[peaks], 'p', label='Detected Peaks')

        true_annotations = np.asarray(tb.annotation_indices)
        plt.plot(true_annotations, signal[true_annotations], 'o', label='True Annotations')

        plt.legend()

        # uncomment line to show the plot
        plt.show()

    # if we wish to save all the stats to a file
    if file_debug:
        # print out more complex stats to the debug file
        debug_file_path = database_name + "_debug_stats.txt"
        debug_file = open(debug_file_path, 'w')

        # print out indices of all false positives
        debug_file.writelines("-----False Positives Indices-----\n")
        for fp in unmatched:
            debug_file.writelines(str(fp) + "\n")

        # print out indices of all false negatives
        debug_file.writelines("-----False Negatives Indices-----\n")
        for fn in remaining:
            debug_file.writelines(str(fn.sample) + "\n")

        # close file that we writing
        debug_file.close()

    if print_debug:
        print("-------------------------------------------------")
        print("Database|\t\tTP|\t\tFP|\t\tFN|\t\tF1")
        print(database_name, "|\t\t", true_positive, "|\t", false_positive, '|\t', false_negative, '|\t', round(f1, 3))
        print("-------------------------------------------------")

    print("Done!")
