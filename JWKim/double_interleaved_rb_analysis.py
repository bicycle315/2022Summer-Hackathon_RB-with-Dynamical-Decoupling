# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Double Interleaved RB analysis class.
"""
from typing import List, Union

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import AnalysisResultData, ExperimentData
from qiskit_experiments.library.randomized_benchmarking import RBAnalysis


class DoubleInterleavedRBAnalysis(RBAnalysis):
    r"""A class to analyze double interleaved randomized benchmarking experiment.
    """

    def __init__(self):
        super().__init__()
        self._num_qubits = None

    __series__ = [
        curve.SeriesDef(
            name="Standard",
            fit_func=lambda x, a, alpha, alpha_c1, alpha_c2, b: curve.fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            filter_kwargs={"interleaved": False, "pos": 0},
            plot_color="red",
            plot_symbol=".",
            model_description=r"a \alpha^{x} + b",
        ),
        curve.SeriesDef(
            name="Interleaved-First",
            fit_func=lambda x, a, alpha, alpha_c1, alpha_c2, b: curve.fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha * alpha_c1, baseline=b
            ),
            filter_kwargs={"interleaved": True, "pos": 1},
            plot_color="orange",
            plot_symbol="^",
            model_description=r"a (\alpha_c1\alpha)^{x} + b",
        ),
        curve.SeriesDef(
            name="Interleaved-Second",
            fit_func=lambda x, a, alpha, alpha_c1, alpha_c2, b: curve.fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha * alpha_c2, baseline=b
            ),
            filter_kwargs={"interleaved": True, "pos": 2},
            plot_color="green",
            plot_symbol="^",
            model_description=r"a (\alpha_c2\alpha)^{x} + b",
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        default_options = super()._default_options()
        default_options.result_parameters = ["alpha", "alpha_c1", "alpha_c2"]
        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(
            a=(0, 1),
            alpha=(0, 1),
            alpha_c1=(0, 1),
            alpha_c2=(0, 1),
            b=(0, 1),
        )

        b_guess = 1 / 2**self._num_qubits
        a_guess = 1 - b_guess

        # for standard RB curve
        std_curve = curve_data.get_subset_of("Standard")
        alpha_std = curve.guess.rb_decay(std_curve.x, std_curve.y, a=a_guess, b=b_guess)

        # for interleaved RB curve
        int_curve1 = curve_data.get_subset_of("Interleaved-First")
        alpha_int1 = curve.guess.rb_decay(int_curve1.x, int_curve1.y, a=a_guess, b=b_guess)
        int_curve2 = curve_data.get_subset_of("Interleaved-Second")
        alpha_int2 = curve.guess.rb_decay(int_curve2.x, int_curve2.y, a=a_guess, b=b_guess)

        alpha_c1 = min(alpha_int1 / alpha_std, 1.0)
        alpha_c2 = min(alpha_int2 / alpha_std, 1.0)

        user_opt.p0.set_if_empty(
            b=b_guess,
            a=a_guess,
            alpha=alpha_std,
            alpha_c1=alpha_c1,
            alpha_c2=alpha_c2,
        )

        return user_opt

    def _format_data(
        self,
        curve_data: curve.CurveData,
    ) -> curve.CurveData:
        """Postprocessing for the processed dataset.

        Args:
            curve_data: Processed dataset created from experiment results.

        Returns:
            Formatted data.
        """
        # TODO Eventually move this to data processor, then create RB data processor.

        # take average over the same x value by keeping sigma
        data_allocation, xdata, ydata, sigma, shots = curve.data_processing.multi_mean_xy_data(
            series=curve_data.data_allocation,
            xdata=curve_data.x,
            ydata=curve_data.y,
            sigma=curve_data.y_err,
            shots=curve_data.shots,
            method="sample",
        )

        # sort by x value in ascending order
        data_allocation, xdata, ydata, sigma, shots = curve.data_processing.data_sort(
            series=data_allocation,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
            shots=shots,
        )

        return curve.CurveData(
            x=xdata,
            y=ydata,
            y_err=sigma,
            shots=shots,
            data_allocation=data_allocation,
            labels=curve_data.labels,
        )

    def _create_analysis_results(
        self,
        fit_data: curve.FitData,
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for important fit parameters.

        Args:
            fit_data: Fit outcome.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """
        outcomes = super()._create_analysis_results(fit_data, quality, **metadata)

        nrb = 2**self._num_qubits
        scale = (nrb - 1) / nrb

        # alpha = fit_data.fitval("alpha")
        alpha_c1 = fit_data.fitval("alpha_c1")
        alpha_c2 = fit_data.fitval("alpha_c2")

        # Calculate epc_est (=r_c^est) - Eq. (4):
        epc1 = scale * (1 - alpha_c1)
        epc2 = scale * (1 - alpha_c2)

        # # Calculate the systematic error bounds - Eq. (5):
        # systematic_err_1 = scale * (abs(alpha.n - alpha_c.n) + (1 - alpha.n))
        # systematic_err_2 = (
        #     2 * (nrb * nrb - 1) * (1 - alpha.n) / (alpha.n * nrb * nrb)
        #     + 4 * (np.sqrt(1 - alpha.n)) * (np.sqrt(nrb * nrb - 1)) / alpha.n
        # )
        #
        # systematic_err = min(systematic_err_1, systematic_err_2)
        # systematic_err_l = epc.n - systematic_err
        # systematic_err_r = epc.n + systematic_err

        outcomes.append(
            AnalysisResultData(
                name="EPC1",
                value=epc1,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra={
                    # "EPC_systematic_err": systematic_err,
                    # "EPC_systematic_bounds": [max(systematic_err_l, 0), systematic_err_r],
                    **metadata,
                },
            )
        )
        outcomes.append(
            AnalysisResultData(
                name="EPC2",
                value=epc2,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra={
                    # "EPC_systematic_err": systematic_err,
                    # "EPC_systematic_bounds": [max(systematic_err_l, 0), systematic_err_r],
                    **metadata,
                },
            )
        )

        return outcomes

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        """Initialize curve analysis with experiment data.

        This method is called ahead of other processing.

        Args:
            experiment_data: Experiment data to analyze.
        """
        super()._initialize(experiment_data)

        # Get qubit number
        self._num_qubits = len(experiment_data.metadata["physical_qubits"])
