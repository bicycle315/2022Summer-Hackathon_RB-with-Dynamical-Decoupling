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
Interleaved RB Experiment class.
"""
from typing import Union, Iterable, Optional, List, Sequence

from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend

from .rb_experiment import StandardRB
from .double_interleaved_rb_analysis import DoubleInterleavedRBAnalysis


class DoubleInterleavedRB(StandardRB):
    """Interleaved randomized benchmarking experiment with two interleaved operations.
    """

    def __init__(
        self,
        first_interleaved_op: Union[QuantumCircuit, Instruction, Clifford],
        second_interleaved_op: Union[QuantumCircuit, Instruction, Clifford],
        qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: bool = False,
    ):
        """Initialize an interleaved randomized benchmarking experiment.

        Args:
            first_interleaved_op: The first operation to interleave,
                    given either as a group element or as an instruction/circuit
            second_interleaved_op: The second operation to interleave,
                    given either as a group element or as an instruction/circuit
            qubits: list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each
                         sequence length
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.
        """
        try:
            self._first_interleaved_elem = (first_interleaved_op, Clifford(first_interleaved_op))
        except QiskitError as error:
            raise QiskitError(
                "Interleaved element {} could not be converted to Clifford element".format(
                    first_interleaved_op.name
                )
            ) from error
        try:
            self._second_interleaved_elem = (second_interleaved_op, Clifford(second_interleaved_op))
        except QiskitError as error:
            raise QiskitError(
                "Interleaved element {} could not be converted to Clifford element".format(
                    second_interleaved_op.name
                )
            ) from error
        super().__init__(
            qubits,
            lengths,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
        )
        self.analysis = DoubleInterleavedRBAnalysis()
        self.analysis.set_options(outcome="0" * self.num_qubits)

    def _sample_circuits(self, lengths, rng):
        circuits = []
        for length in lengths if self._full_sampling else [lengths[-1]]:
            elements = self._clifford_utils.random_clifford_circuits(self.num_qubits, length, rng)
            element_lengths = [len(elements)] if self._full_sampling else lengths
            std_circuits = self._generate_circuit(elements, element_lengths)
            for circuit in std_circuits:
                circuit.metadata["interleaved"] = False
                circuit.metadata["pos"] = 0
            circuits += std_circuits

            int_elements = []
            for element in elements:
                int_elements.append(element)
                int_elements.append(self._first_interleaved_elem)

            int_elements_lengths = [length * 2 for length in element_lengths]
            int_circuits = self._generate_circuit(int_elements, int_elements_lengths)
            for circuit in int_circuits:
                circuit.metadata["interleaved"] = True
                circuit.metadata["pos"] = 1
                circuit.metadata["xval"] = circuit.metadata["xval"] // 2
            circuits += int_circuits

            int_elements = []
            for element in elements:
                int_elements.append(element)
                int_elements.append(self._second_interleaved_elem)

            int_elements_lengths = [length * 2 for length in element_lengths]
            int_circuits = self._generate_circuit(int_elements, int_elements_lengths)
            for circuit in int_circuits:
                circuit.metadata["interleaved"] = True
                circuit.metadata["pos"] = 2
                circuit.metadata["xval"] = circuit.metadata["xval"] // 2
            circuits += int_circuits

        return circuits
