import os
from pathlib import Path

import pandas as pd

# def read_data(path):
#     """Read tab-separated AquiMod data."""
#     return pd.read_csv(path, sep="\t", index_col=False, engine="python")


# def edit_input_file(path, line_number, text):
#     """Edit a line in Input.txt"""
#     with open(path, "r") as f:
#         lines = f.readlines()
#         lines[line_number] = text
#     with open(path, "w") as f:
#         f.writelines(lines)


# def read_parameter_lims(paths: list[str]):
#     """
#     Get the parameter space from Calibration folder and store it in a dictionary with
#     parameter name as key and list of min and max as value.
#     """
#     output = {}
#     for path in paths:
#         with open(path, "r") as f:
#             lines = f.readlines()
#             for i, line in enumerate(lines):
#                 line = line.replace("\n", "")
#                 if i % 3 == 0:
#                     param = line
#                 elif i % 3 == 1:
#                     vals = [float(val) for val in line.split(" ")]
#                 output[param] = vals
#     return output


# def write_eval_file(df_dict: dict[str, pd.DataFrame]):
#     """
#     Function to write data to calibration files.
#     df_dict is a dictionary of module names and the dataframes to be saved.
#     """
#     for module, df in df_dict.items():
#         path = Path()
#         df.to_csv(path, sep="\t", index=False)


# def run_aquimod(model_dir, sim_mode=None, num_runs=None):
#     """
#     Execute AquiModAWS with chosen model directory.
#     Only number of runs can be modified.
#     Returns calibration output as single dataframe.
#     """
#     # Delete previous contents of output folder
#     for p in Path(model_dir, "Output").glob("*"):
#         os.remove(p)

#     input_path = Path(model_dir, "Input.txt")

#     # Set simulation mode if specified
#     if sim_mode is not None:
#         edit_input_file(input_path, 4, f"{sim_mode}\n")

#     # Set number of runs if specfied
#     if num_runs is not None:
#         edit_input_file(input_path, 10, f"{num_runs}\n")

#     # Run AquiModAWS
#     os.system(f"AquiModAWS {model_dir}")

#     # Read outputs
#     output_list = [read_data(p) for p in Path(model_dir, "Output").glob("*")]
#     return pd.concat(output_list, axis=1)


class AquiModAWS:
    def __init__(self, model_dir, parameter_path="parameter_data.csv"):
        self.model_dir = model_dir
        self.input_path = Path(model_dir, "Input.txt")
        self.observations_path = Path(model_dir, "Observations.txt")
        self.parameter_data = pd.read_csv(parameter_path)
        self.module_config = self.get_module_config()
        self.calibration_paths = self._get_calibration_paths()
        self.evaluation_paths = self._get_evaluation_paths()

    def _delete_dir_contents(self, directory):
        for path in Path(self.model_dir, directory).glob("*"):
            os.remove(path)

    def _read_data(self, path):
        """Read tab-separated data."""
        return pd.read_csv(path, sep="\t", index_col=False, engine="python")

    def _get_calibration_paths(self) -> Path:
        """Create paths to calibration files using module configuration"""
        return [
            Path(
                self.model_dir,
                "Calibration",
                self.parameter_data[
                    self.parameter_data["module_number"]
                    == self.module_config[module_number],
                    "module_name",
                ].unique()[0]
                + "_calib.txt",
            )
            for module_number in self.module_config
        ]

    def _get_evaluation_paths(self) -> Path:
        """Create paths to evaluation files using module configuration"""
        return [
            Path(
                self.model_dir,
                "Evaluation",
                self.parameter_data[
                    self.parameter_data["module_number"]
                    == self.module_config[module_number],
                    "module_name",
                ].unique()[0]
                + "_eval.txt",
            )
            for module_number in self.module_config
        ]

    def get_outputs(self):
        output_list = [read_data(p) for p in Path(self.model_dir, "Output").glob("*")]
        return pd.concat(output_list, axis=1)

    def edit_input_file(self, line_number, text):
        """Edit a line in Input.txt"""
        with open(self.input_path, "r") as f:
            lines = f.readlines()
            lines[line_number] = text
        with open(self.input_path, "w") as f:
            f.writelines(lines)

    def _get_module_config(self) -> list[int]:
        """Get module config from input file and return as list of integers"""
        with open(self.input_path, "r") as f:
            lines = f.readlines()
        lines = lines[1].replace("\n", "").split(" ")
        return [int(val) for val in lines]

    def read_parameter_lims(self):
        """
        Read parameter limits from calibration file and save into self.parameter_data.
        """
        # Get module name from path
        for path in self.calibration_paths:
            module = str(path.stem).split("_")[0]

            # Open calibration file and read lines
            with open(path, "r") as f:
                lines = f.readlines()
                lines = [line.replace("\n", "") for line in lines]

            # Loop through every line containing min-max data
            for i, line in enumerate(lines):
                if i % 3 != 1:
                    continue
                # Assign min-max values and parameter name to variables
                minimum, maximum = [float(val) for val in line.split(" ")]
                parameter = lines[i - 1]

                # Assign min and max to variables to specific cell in parameter_data
                self.parameter_data[
                    self.parameter_data["module_name"]
                    == module & self.parameter_data["parameter"]
                    == parameter,
                    "min",
                ] = minimum
                self.parameter_data[
                    self.parameter_data["module_name"]
                    == module & self.parameter_data["parameter"]
                    == parameter,
                    "max",
                ] = maximum

    def write_eval_file(
        self, soil: pd.DataFrame, unsaturated: pd.DataFrame, saturated: pd.DataFrame
    ):
        """Write data to evaluation files."""
        soil.to_csv(self.evaluation_paths[0], sep="\t", index=False)
        unsaturated.to_csv(self.evaluation_paths[1], sep="\t", index=False)
        saturated.to_csv(self.evaluation_paths[2], sep="\t", index=False)

    def run(self, sim_mode=None, num_runs=None):
        # Set simulation mode if specified
        if sim_mode is not None:
            self.edit_input_file(4, f"{sim_mode}\n")
        # Set number of runs if specfied
        if num_runs is not None:
            self.edit_input_file(10, f"{num_runs}\n")

        # Delete previous contents of output folder
        self._delete_dir_contents("Output")
        # Run AquiModAWS
        os.system(f"AquiModAWS {self.model_dir}")
        # Read outputs
        return self.read_outputs()
