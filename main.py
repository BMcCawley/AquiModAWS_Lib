import datetime
import time

import pandas as pd

from autocal.aquimod import AquiModAWS

# This calibration set up looks best so far


def main():
    model = AquiModAWS("model")
    calibration = model.calibrate(
        num_complxes=5, complx_size=10, simplx_size=5, alpha=20, num_cycles=10
    )
    calibration_df = pd.concat(calibration.values(), axis=1)
    calibration_df.to_csv("calibration.csv")
    print(calibration_df["ObjectiveFunction"].head())


if __name__ == "__main__":
    main()
    print(f"Completed in {datetime.timedelta(seconds=round(time.perf_counter()))}")
