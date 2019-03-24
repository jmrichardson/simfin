
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()


from tsfresh import extract_features
extracted_features = extract_features(timeseries, column_id="id", column_sort="time")


impute(extracted_features)
features_filtered = select_features(extracted_features, y)
















