import datetime
import time

import pandas as pd

from autocal.aquimod import AquiModAWS


def main():
    model = AquiModAWS("model")
    calibration = model.calibrate(
        num_complxes=1, complx_size=100, simplx_size=10, alpha=20, num_cycles=100
    )
    calibration_df = pd.concat(calibration.values(), axis=1)
    calibration_df.to_csv("calibration.csv")
    print(calibration_df["ObjectiveFunction"].head())


if __name__ == "__main__":
    main()
    print(f"Completed in {datetime.timedelta(seconds=round(time.perf_counter()))}")
