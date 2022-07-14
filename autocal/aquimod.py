import os
from pathlib import Path

import pandas as pd


class AquiModAWS:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.input_path = Path(model_dir, "Input.txt")
        self.observations_path = Path(model_dir, "Observations.txt")
        self._model_data = pd.read_csv("model_data.csv")

    def _edit_line(self, path, line_number: int, text: str):
        """Edit a text file at a certain line. Automatically places newline char."""
        with open(path, "r") as f:
            lines = f.readlines()
            lines[line_number] = text + "\n"
        with open(self.input_path, "w") as f:
            f.writelines(lines)

    def _read_line(self, path, line_number: int) -> str:
        """Read contents of a certain line within a file"""
        with open(path, "r") as f:
            lines = f.readlines()
        return lines[line_number].replace("\n", "")

    def _read_data(self, path, skiprows=None) -> pd.DataFrame:
        """Read AquiMod data in a table format"""
        return pd.read_csv(path, sep="\t", index_col=False, skiprows=skiprows)

    def _delete_dir_contents(self, directory):
        """Delete contents"""
        for path in Path(self.model_dir, directory).glob("*"):
            os.remove(path)

    @property
    def module_config(self) -> dict[str, int]:
        """Get module numbers from input file"""
        line = self._read_line(self.input_path, 1).split(" ")
        return {
            "soil": int(line[0]),
            "unstaurated": int(line[1]),
            "saturated": int(line[2]),
        }

    @module_config.setter
    def module_config(self, config: list[int]):
        """Set module numbers in input file"""
        text = " ".join([str(val) for val in config])
        self._edit_line(self.input_path, 1, text)

    @property
    def simulation_mode(self) -> str:
        """Get the simulation mode from the input file"""
        return self._read_line(self.input_path, 4)

    @simulation_mode.setter
    def simulation_mode(self, mode: str):
        """Set simulation mode in input file"""
        self._edit_line(self.input_path, 4, mode)

    @property
    def number_of_runs(self) -> int:
        """Get number of runs from input file"""
        return self._read_line(self.input_path, 10)

    @number_of_runs.setter
    def number_of_runs(self, num_runs: int):
        """Set number of runs in input file"""
        self._edit_line(self.input_path, 10, str(num_runs))

    @property
    def calibrated_variable(self) -> str:
        """Get calibrated variable from input file"""
        self._read_line(self.input_path, 7)

    @calibrated_variable.setter
    def calibrated_variable(self, variable: str):
        """Set calibrated variable (either 'g' or 's') in input file"""
        self._edit_line(self.input_path, 7, variable)

    @property
    def module_names(self) -> dict[str, str]:
        """Get module names"""
        df_dict = {
            "soil": self._model_data[self._model_data["component"] == "soil"],
            "unsaturated": self._model_data[
                self._model_data["component"] == "unsaturated"
            ],
            "saturated": self._model_data[self._model_data["component"] == "saturated"],
        }
        modules = self.module_config

        return {
            component: df[
                df["module_number"] == modules[component], "module_name"
            ].unique()[0]
            for component, df in df_dict.items()
        }

    @property
    def parameters(self) -> dict[str, list[str]]:
        """Get parameter names as list value for each component key"""
        return {
            module_name: list(
                self._model_data[
                    self._model_data["module_name"] == module_name, "component_name"
                ].unique()
            )
            for module_name in self.module_names.values()
        }

    @property
    def parameter_line_numbers(self) -> dict[str, int]:
        """Get parameter line numbers as a dictionary with parameter as key"""
        return {
            parameter: self._model_data[
                (self._model_data["module_name"] == module)
                & (self._model_data["parameter"] == parameter),
                "line_number",
            ].unique()[0]
            for module, parameter in self.parameters.items()
        }

    @property
    def calibration_paths(self) -> dict[str, Path]:
        """Get paths to calibration files"""
        return {
            component: Path(self.model_dir, "Calibration", module_name + "_calib.txt")
            for component, module_name in self.module_names
        }

    @property
    def evaluation_paths(self) -> dict[str, Path]:
        """Get paths to evaluation files"""
        return {
            component: Path(self.model_dir, "Evaluation", module_name + "_eval.txt")
            for component, module_name in self.module_names
        }

    @property
    def output_calibration_paths(self) -> dict[str, Path]:
        """Get the paths to output calibration files"""
        paths = {
            component: Path(self.model_dir, "Output", module + "_calib.out")
            for component, module in self.module_names.items()
        }
        if self.calibrated_variable == "g":
            paths["fit"] = Path(self.model_dir, "Output", "fit_calib_GWL.out")
        elif self.calibrated_variable == "s":
            paths["fit"] = Path(self.model_dir, "Output", "fit_calib_SM.out")

        return paths

    @property
    def output_evaluation_paths(self) -> dict[str, list[Path]]:
        """Get the paths to output evaluation files"""
        # Could search for paths using path.glob or use the number of runs property
        # Opted for using the number of runs property
        # Also need to consider if some output files have not been saved, as with "write model output files"
        # For now assume that everything is saved
        path_dict = {}
        for component, module in self.module_names.items():
            path_list = [
                Path(self.model_dir, "Output", module + f"_TimeSeries{i}.out")
                for i in range(1, self.number_of_runs + 1)
            ]
            path_dict[component] = path_list

        if self.calibrated_variable == "g":
            path_dict["fit"] = Path(self.model_dir, "Output", "fit_eval_GWL.out")
        elif self.calibrated_variable == "s":
            path_dict["fit"] = Path(self.model_dir, "Output", "fit_eval_SM.out")

    @property
    def calibration_parameters(self) -> dict[str, pd.DataFrame]:
        """
        Get parameter limits as dict of dataframes.
        Dataframes arranged with parameters as columns.
        """
        # Instantiate outer dictionary
        outer = {}
        # Loop through each component and its calibration file path
        for component, path in self.calibration_paths.items():
            # Instantiate inner dictionary
            inner = {}
            # Loop through each parameter in the current component
            for parameter in self.parameters[component]:
                # Get min and max value for each parameter as string
                minmax = self._read_line(path, self._parameter_line_numbers[parameter])
                # Split string into list and cast elements into floats
                minmax_list = [float(val) for val in minmax.split(" ")]
                # Assign list of min and max floats to inner dictionary
                inner[parameter] = minmax_list
            # Convert inner dictionary to DataFrame and assign to outer dictionary
            outer[component] = pd.DataFrame(inner, index=["min", "max"])

        return outer

    @calibration_parameters.setter
    def calibration_parameters(self, calib_dict: dict[str, pd.DataFrame]) -> None:
        """Set calibration parameters"""
        # Don't think I need to work on this one right now
        pass

    @property
    def evaluation_parameters(self) -> dict[str, pd.DataFrame]:
        """Get evaluation parameters as dict of dataframes"""
        return {
            component: self._read_data(path)
            for component, path in self.evaluation_paths.items()
        }

    @evaluation_parameters.setter
    def evaluation_parameters(self, eval_dict: dict[str, pd.DataFrame]) -> None:
        """Set evaluation parameters"""
        for component, df in eval_dict.items():
            df.to_csv(self.evaluation_paths[component], sep="\t", index=False)

    def run(
        self,
        module_config: list[int] = None,
        sim_mode: str = None,
        calib_var: str = None,
        num_runs: int = None,
    ):
        if module_config is not None:
            self.module_config = module_config
        if sim_mode is not None:
            self.simulation_mode = sim_mode
        if calib_var is not None:
            self.calibrated_variable = calib_var
        if num_runs is not None:
            self.number_of_runs = num_runs

        # self._delete_dir_contents(Path(self.model_dir, "Output"))
        os.system(f"AquiModAWS {self.model_dir}")

    def read_performance_output(self) -> dict[str, pd.DataFrame]:
        """
        Read output files from the most recent call of self.run().
        Includes parameter values and model performance.
        """
        # Don't forget that a user can just look at the output files themselves
        # GWLForecaster reads timeseries outputs and creates a csv with a column for
        # each individual model run.
        # GWLForecaster isn't bothered at all with fit as it is just forecast data.
        if self.simulation_mode == "c":
            return {
                component: self._read_data(path)
                for component, path in self.output_calibration_paths.items()
            }
        elif self.simulation_mode == "e":
            output = {
                component: self._read_data(path)
                for component, path in self.evaluation_paths.items()
            }
            output["fit"] = pd.read_csv(
                self.output_evaluation_paths["fit"], sep="\t", index_col=False
            )
            return output

    def read_timeseries_output(self):
        """Read output timeseries files"""
        pass
