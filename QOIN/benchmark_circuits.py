#!/usr/bin/env python
# coding: utf-8


from util_imports import *
from Abstract_Interface import *
import random
from qiskit.circuit.library import Permutation, QFT
import logging
import math
import textwrap

import pandas as pd
from qiskit import Aer
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute
from qiskit.circuit.add_control import add_control
from qiskit.compiler import transpile
from qiskit.extensions import *
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import plot_histogram

logger = logging.getLogger(__name__)


class StringComparator:

    def __init__(self, target, db, symbol_length=1, symbol_count=None, is_binary=True,
                 shots=8192, quantum_instance=Aer.get_backend('qasm_simulator'),
                 optimize_for=None, optimization_levels=None, attempts_per_optimization_level=None,
                 default_dataset=False, t=1, p_pqm=False
                 ):
        """
        Compare a string against a set of strings.

        :param target: target string
        :param db: a set of strings (passed in a list) against which we compare the target string
        :param symbol_length: the number of characters that codify a symbol; used only when is_binary == True
        :param symbol_count: the number of characters in the alphabet; used only when is_binary == False.
               the default value is None -- in this case the number of symbols is determined automatically based on the
               number of distinct characters in the `db`. However, we may need to override this number for machine
               learning tasks as a particular dataset may not have all the characters present in all the classes
        :param is_binary: are we dealing with binary strings?
        :param shots: the number of measurements to take
        :param quantum_instance: the pointer to the backend on which we want to execute the code
        :param optimize_for: architecture for which the code should be optimized, e.g., `FakeMontreal()`.
                             If none -- no optimization will be performed.
        :param optimization_levels: a list of optimization levels (QisKit transpiler takes values between 0 and 3)
        :param attempts_per_optimization_level: a number of times transpiler will be executed to find optimal circuit
        :param default_dataset: When True, this enables creation of a database containing all strings in equal superpositon.
                                When False, the database is initialized with strings passed in parameter `db`.
        :param p_pqm: When True, this will run the storage and retrieval algorithms of parametric probabilistic quantum memory
                      When False, this will run the extended p-pqm storage and retrieval algorithms
        :param t: parameter `t, a value within `(0, 1]` range is used by P-PQM algorithm to compute weighted Hamming distance,
                  which may improve performance of machine learning classification.
                  When `t=1` (the default value), P-PQM reduces to PQM.
        """
        self.t = t
        self.quantum_instance = quantum_instance
        self.shots = shots
        self.is_binary = is_binary
        self.default_dataset = default_dataset
        self.p_pqm = p_pqm

        if is_binary:  # check that strings contain only 0s and 1s
            self.symbol_length = symbol_length
            self.target_string, self.string_db = self._massage_binary_strings(target, db)
        else:
            self.target_string, self.string_db, self.symbol_length, self.symb_map = \
                self._massage_symbol_strings(target, db, symbol_count)

        self.input_size = len(self.target_string)

        logger.debug(f"Target string is '{self.target_string}'")
        logger.debug(f"Database is {self.string_db}")

        # Create Circuit
        if p_pqm:
            self.u_register_len = 2
            self.u_register = QuantumRegister(self.u_register_len)
            self.memory_register = QuantumRegister(self.input_size)
            self.pattern_register = QuantumRegister(self.input_size)
            self.qubits_range = self.u_register_len + self.input_size
            self.classic_register = ClassicalRegister(self.input_size + 1)

            self.circuit = QuantumCircuit(self.u_register, self.memory_register, self.pattern_register,
                                          self.classic_register)
            self._store_information_p_pqm(self.string_db)
            self._retrieve_information_P_PQM(self.target_string)

            self.circuit.measure(range(1, self.qubits_range), range(0, self.input_size + 1))

        else:
            self.u_register_len = 2
            self.u_register = QuantumRegister(self.u_register_len)
            self.memory_register = QuantumRegister(self.input_size)
            self.size_of_single_ham_register = math.floor(self.input_size / self.symbol_length)
            self.single_ham_dist_register = QuantumRegister(self.size_of_single_ham_register)
            self.qubits_range = self.size_of_single_ham_register + self.u_register_len + self.input_size
            self.classic_register = ClassicalRegister(self.qubits_range - self.size_of_single_ham_register - 1)
            self.circuit = QuantumCircuit(self.u_register, self.memory_register, self.single_ham_dist_register,
                                          self.classic_register)

            # TODO: we can use the attribute directly and not pass it as a parameter in the next two function calls

            if default_dataset:
                self._store_default_database(len(self.string_db[0]))
            if not default_dataset:
                self._store_information(self.string_db)

            self._retrieve_information(self.target_string)

            self.circuit.measure(range(1, self.qubits_range - self.size_of_single_ham_register),
                                 range(0, self.qubits_range - self.size_of_single_ham_register - 1))

        if optimize_for is not None:
            self._optimize_circuit(optimize_for, optimization_levels=optimization_levels,
                                   attempts_per_optimization_level=attempts_per_optimization_level)

        self.results = None

    def _optimize_circuit(self, backend_architecture, optimization_levels=range(0, 1),
                          attempts_per_optimization_level=1):
        """
        Try to optimize circuit by minimizing it's depth. Currently, it does a naive grid search.

        :param backend_architecture: the architecture for which the code should be optimized
        :param optimization_levels: a range of optimization levels
                                    (the transpiler currently supports the values between 0 and 3)
        :param attempts_per_optimization_level: the number of attempts per optimization level
        :return: None
        """
        cfg = backend_architecture.configuration()

        best_depth = math.inf
        best_circuit = None

        depth_stats = []

        for opt_level in optimization_levels:
            for attempt in range(0, attempts_per_optimization_level):
                optimized_circuit = transpile(self.circuit, coupling_map=cfg.coupling_map, basis_gates=cfg.basis_gates,
                                              optimization_level=opt_level) #, layout_method='sabre',
                                              # routing_method='sabre')
                current_depth = optimized_circuit.depth()
                depth_stats.append([opt_level, current_depth])
                if current_depth < best_depth:
                    best_circuit = optimized_circuit
                    best_depth = current_depth
        self.circuit = best_circuit

        self.optimizer_stats = pd.DataFrame(depth_stats, columns=['optimization_level', 'circuit_depth']).\
            groupby('optimization_level').describe().unstack(1).reset_index().\
            pivot(index='optimization_level', values=0, columns='level_1')
        logger.debug(f"Optimized depth is {self.circuit.depth()}")

    def get_optimizer_stats(self):
        """
        Get optimizer stats

        :return: pandas data frame with the optimizer stats
        """
        try:
            return self.optimizer_stats
        except AttributeError:
            raise AttributeError("Optimizer was not invoked, no stats present")

    def _massage_binary_strings(self, target, db):
        """
        Massage binary strings and perform sanity checks

        :param target: target string
        :param db: database of strings
        :return: massaged target and database strings
        """
        # sanity checks
        if not isinstance(target, str):
            raise TypeError("Target string should be of type str")
        for my_str in db:
            if not isinstance(my_str, str):
                raise TypeError(f"Database string {my_str} should be of type str")

        bits_in_str_cnt = len(target)
        symbols_in_str_cnt = bits_in_str_cnt / self.symbol_length
        if bits_in_str_cnt % symbols_in_str_cnt != 0:
            raise TypeError(f"Possible data corruption: bit_count MOD symbol_length should be 0, but got "
                            f"{bits_in_str_cnt % symbols_in_str_cnt}")

        for my_str in db:
            if len(my_str) != bits_in_str_cnt:
                raise TypeError(
                    f"Target string size is {bits_in_str_cnt}, but db string {my_str} size is {len(my_str)}")

        if not self.is_str_binary(target):
            raise TypeError(
                f"Target string should be binary, but the string {target} has these characters {set(target)}")

        for my_str in db:
            if not self.is_str_binary(my_str):
                raise TypeError(f"Strings in the database should be binary, but the string {my_str} "
                                f"has these characters {set(my_str)}")

        return target, db

    @staticmethod
    def _massage_symbol_strings(target, db, override_symbol_count=None):
        """
        Massage binary strings and perform sanity checks

        :param target: target string
        :param db: database of strings
        :param override_symbol_count: number of symbols in the alphabet, if None -- determined automatically
        :return: target string converted to binary format,
                 database strings converted to binary format,
                 length of symbol in binary format,
                 map of textual symbols to their binary representation (used only for debugging)
        """

        # sanity checks
        if not isinstance(target, list):
            raise TypeError("Target string should be of type list")
        for my_str in db:
            if not isinstance(my_str, list):
                raise TypeError(f"Database string {my_str} should be of type list")

        # compute  strings' length
        symbols_in_str_cnt = len(target)
        for my_str in db:
            if len(my_str) != symbols_in_str_cnt:
                raise TypeError(
                    f"Target string has {symbols_in_str_cnt} symbols, but db string {my_str} has {len(my_str)}")

        # get distinct symbols
        symbols = {}
        id_cnt = 0
        for symbol in target:
            if symbol not in symbols:
                symbols[symbol] = id_cnt
                id_cnt += 1
        for my_str in db:
            for symbol in my_str:
                if symbol not in symbols:
                    symbols[symbol] = id_cnt
                    id_cnt += 1

        # override symbol length if symbol count was specified by the user
        dic_symbol_count = len(symbols)
        if override_symbol_count is not None:
            if dic_symbol_count > override_symbol_count:
                raise ValueError(f"Alphabet has at least {dic_symbol_count}, "
                                 f"but the user asked only for {override_symbol_count} symbols")
            dic_symbol_count = override_symbol_count

        # figure out how many bits a symbol needs
        symbol_length = math.ceil(math.log2(dic_symbol_count))
        logger.debug(f"We got {dic_symbol_count} distinct symbols requiring {symbol_length} bits per symbol")

        # convert ids for the symbols to binary strings
        bin_format = f"0{symbol_length}b"
        for symbol in symbols:
            symbols[symbol] = format(symbols[symbol], bin_format)

        # now let's produce binary strings
        # TODO: += is not the most efficient way to concatenate strings, think of a better way
        target_bin = ""
        for symbol in target:
            target_bin += symbols[symbol]

        db_bin = []
        for my_str in db:
            db_str_bin = ""
            for symbol in my_str:
                db_str_bin += symbols[symbol]
            db_bin.append(db_str_bin)

        return target_bin, db_bin, symbol_length, symbols

    def run(self, quantum_instance=None):
        """
        Execute the circuit and return a data structure with details of the results

        :param quantum_instance: the pointer to the backend on which we want to execute the code
               (overwrites the backend specified in the constructor)
        :return: a dictionary containing hamming distance and p-values for each string in the database,
                 along with extra debug info (raw frequency count and the probability of measuring
                 register c as 0)
        """
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance

        # 1) 展开自定义/受控门，避免嵌套
        flat = self.circuit.decompose(reps=10)

        # 2) 分解到 QASM2 支持的基门
        qasm2_basis = [
            'u','p','cx','sx','sxdg','x','y','z','h','s','sdg','t','tdg',
            'rz','ry','rx','swap','cz','cy','ch','ccx','barrier','measure','reset'
        ]
        flat = transpile(flat, basis_gates=qasm2_basis, optimization_level=0)
        
        result = self.quantum_instance.execute(flat)
        results_raw = result.get_counts(flat)

        # tweak raw results and add those strings that have 0 shots/pulses associated with them
        # these are the strings that will have hamming distance equal to the total number of symbols
        for string in self.string_db:
            full_binary_string = string[::-1] + "1"
            if full_binary_string not in results_raw:
                results_raw[full_binary_string] = 0

        # Massage results
        count_dic, useful_shots_count = self._get_count_of_useful_values(results_raw)
        p_values = []

        for my_str in self.string_db:
            p_values.append(count_dic[my_str] / useful_shots_count)

        probability_of_measuring_register_c_as_0 = float(sum(p_values))
        # re-normalize p-values, so that they sum up to 1.0
        if probability_of_measuring_register_c_as_0 != 0:  # else all values are zero anyway
            for ind in range(len(p_values)):
                p_values[ind] = p_values[ind] / probability_of_measuring_register_c_as_0

        ham_distances = self._convert_p_value_to_hamming_distance(p_values, probability_of_measuring_register_c_as_0)
        self.results = {'p_values': p_values,
                        'hamming_distances': ham_distances,
                        'prob_of_measuring_register_c_as_0': probability_of_measuring_register_c_as_0,
                        'raw_results': results_raw,
                        'useful_shots_count': useful_shots_count
                        }
        return self.results

    def get_circuit_depth(self):
        """
        Get circuit depth
        :return: circuit's depth
        """
        return self.circuit.depth()

    def get_transpiled_circuit_depth(self):
        """
        Get transpiled circuit depth
        :return: circuit's depth
        """
        return self.circuit.decompose().depth()

    def visualise_circuit(self, file_name):
        """
        Visualise circuit

        :param file_name: The name of the file to save the circuit to
        :return: None
        """
        self.circuit.draw(output='mpl', filename=file_name)

    def visualise_transpiled_circuit(self, file_name):
        """
        Visualise transpiled circuit

        :param file_name: The name of the file to save the circuit to
        :return: None
        """
        self.circuit.decompose().draw(output='mpl', filename=file_name)

    def debug_print_raw_shots(self):
        """
        Print raw pulse counts
        :return: None
        """
        print("Raw results")
        print(self.results['raw_results'])

    def debug_produce_histogram(self):
        """
        Generate histogram of raw pulse counts
        :return: None
        """
        print("Histogram")
        plot_histogram(self.results['raw_results'])

    def debug_produce_summary_stats(self):
        """
        Produce summary stats and print it
        :return: summary stats Pandas DataFrame
        """
        print("Summary stats")
        print(f"The number of useful shots is {self.results['useful_shots_count']} out of {self.shots}")
        # compute expected hamming distance
        string_db_expected_hd = []
        for my_str in self.string_db:
            string_db_expected_hd.append(
                self.hamming_distance(self.target_string, my_str, symbol_length=self.symbol_length))
        actual_vs_expected = self._test_output(self.string_db, string_db_expected_hd)
        return actual_vs_expected

    @staticmethod
    def is_str_binary(my_str):
        """
        Check if a string contains only 0s and 1s

        :param my_str: string to check
        :return: True if binary, False -- otherwise
        """
        my_chars = set(my_str)
        if my_chars.issubset({'0', '1'}):
            return True
        else:
            return False

    def _get_count_of_useful_values(self, raw_results):
        """
        Get count of the strings present in the database and the useful number of shots

        :param raw_results: dictionary of registries and count of measurements
        :return: a dictionary of counts, number of useful shots
        """
        p_val_dic = {}
        suffix_length = 1
        useful_shots_count = 0
        for registry_value in raw_results:
            # assume that if the last two bits are set to `00` -- then we measure the degree of closeness
            # and are interested in this observation
            suffix = registry_value[-suffix_length:]

            # extract the middle of the string, which represents the original input
            input_string = registry_value[:-suffix_length]
            # it seems that the values of the strings are stored backward -- inverting
            input_string = input_string[::-1]

            # retain only the strings that were in the database
            # the rest are returned by the actual QC due to noise
            if input_string in self.string_db:
                input_string_cnt = raw_results[registry_value]
                useful_shots_count += input_string_cnt
                if suffix == '1':
                    p_val_dic[input_string] = input_string_cnt

        logging.debug(f"The useful number of shots is {useful_shots_count} out of {self.shots}")
        return p_val_dic, useful_shots_count

    def _store_information(self, logs):
        # Set up initial state
        self.circuit.x(self.u_register[1])
        for my_reg in range(self.size_of_single_ham_register):
            self.circuit.x(self.single_ham_dist_register[my_reg])

        # Load logs into memory register
        for ind in range(len(logs)):
            log = logs[ind]

            self._copy_pattern_to_memory_register(log)
            self.circuit.mct(self.memory_register, self.u_register[0])
            _x = len(logs) + 1 - (ind + 1)
            cs = Operator([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, math.sqrt((_x - 1) / _x), 1 / (math.sqrt(_x))],
                [0, 0, -1 / (math.sqrt(_x)), math.sqrt((_x - 1) / _x)]
            ])
            self.circuit.unitary(cs, [1, 0], label='cs')

            # Reverse previous operations
            self.circuit.mct(self.memory_register, self.u_register[0])
            self._copy_pattern_to_memory_register(log)

    def _store_information_p_pqm(self, logs):
        self.circuit.x(self.u_register[1])
        for i in range(len(logs)):
            string = logs[i]
            logging.debug(f"Processing {string}")
            j = len(string) - 1
            while (j >= 0):
                if (string[j] == '1'):
                    self.circuit.x(self.pattern_register[j])
                j -= 1

            for j in range(self.input_size):
                self.circuit.ccx(self.pattern_register[j], self.u_register[1], self.memory_register[j])

            for j in range(self.input_size):
                self.circuit.cx(self.pattern_register[j], self.memory_register[j])
                self.circuit.x(self.memory_register[j])

            self.circuit.mct(self.memory_register, self.u_register[0])

            x = len(logs) + 1 - (i + 1)
            cs = Operator([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, math.sqrt((x - 1) / x), 1 / (math.sqrt(x))],
                [0, 0, -1 / (math.sqrt(x)), math.sqrt((x - 1) / x)]
            ])

            self.circuit.unitary(cs, [1, 0], label='cs')

            self.circuit.mct(self.memory_register, self.u_register[0])

            for j in range(self.input_size):
                self.circuit.cx(self.pattern_register[j], self.memory_register[j])
                self.circuit.x(self.memory_register[j])

            for j in range(self.input_size):
                self.circuit.ccx(self.pattern_register[j], self.u_register[1], self.memory_register[j])

            j = len(string) - 1
            while (j >= 0):
                if (string[j] == '1'):
                    self.circuit.x(self.pattern_register[j])
                j -= 1

    def _store_default_database(self, length):
        for j in range(length):
            self.circuit.h(self.memory_register[j])
        for j in range(self.size_of_single_ham_register):
            self.circuit.x(self.single_ham_dist_register[j])

    def _fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(self, my_string):
        if not self.p_pqm:
            for j in range(self.input_size):
                if my_string[j] == "0":
                    self.circuit.x(self.memory_register[j])
        else:
            for j in range(self.input_size):
                if my_string[j] == "1":
                    self.circuit.x(self.memory_register[j])

    def _copy_pattern_to_memory_register(self, my_string):
        for j in range(len(my_string)):
            if my_string[j] == "1":
                self.circuit.cx(self.u_register[1], self.memory_register[j])
            else:
                self.circuit.x(self.memory_register[j])

    def _compare_input_and_pattern_for_single_ham_register(self):
        for j in range(self.size_of_single_ham_register):
            idx = self.symbol_length * j
            temp = []
            for ind in range(idx, idx + self.symbol_length):
                temp.append(ind + 2)
            self.circuit.mct(temp, self.single_ham_dist_register[j])

    def _retrieve_information(self, input_string):
        self.circuit.h(1)
        self._fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(input_string)
        self._compare_input_and_pattern_for_single_ham_register()
        u_gate = Operator([
            [math.e ** (complex(0, 1) * math.pi / (2 * ((self.input_size/self.symbol_length) * self.t))), 0],
            [0, 1]
        ])
        for ind in range(self.size_of_single_ham_register):
            self.circuit.unitary(u_gate, self.single_ham_dist_register[ind], label='U')
        u_minus_2_gate = Operator([
            [1 / math.e ** (complex(0, 1) * math.pi / ((self.input_size/self.symbol_length) * self.t)), 0],
            [0, 1]
        ])
        gate2x2 = UnitaryGate(u_minus_2_gate)
        gate2x2_ctrl = add_control(gate2x2, 1, 'CU2x2', '1')
        for j in range(self.size_of_single_ham_register):
            self.circuit.append(gate2x2_ctrl, [1, self.single_ham_dist_register[j]])

        # Reverse previous operations
        self._compare_input_and_pattern_for_single_ham_register()
        self._fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(input_string)

        self.circuit.h(1)

    def _retrieve_information_P_PQM(self, input_string):
        self.circuit.h(1)
        self._fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(input_string)

        u_gate = Operator([
            [math.e ** (complex(0, 1) * math.pi / (2 * ((self.input_size / self.symbol_length) * self.t))), 0],
            [0, 1]
        ])
        for ind in range(self.input_size):
            self.circuit.unitary(u_gate, self.memory_register[ind], label='U')
        u_minus_2_gate = Operator([
            [1 / math.e ** (complex(0, 1) * math.pi / ((self.input_size / self.symbol_length) * self.t)), 0],
            [0, 1]
        ])
        gate2x2 = UnitaryGate(u_minus_2_gate)
        gate2x2_ctrl = add_control(gate2x2, 1, 'CU2x2', '1')
        for j in range(self.input_size):
            self.circuit.append(gate2x2_ctrl, [1, self.memory_register[j]])

        # Reverse previous operations
        self._fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(input_string)

        self.circuit.h(1)


    @staticmethod
    def hamming_distance(str_one, str_two, symbol_length=1):
        """
        Compute hamming distance assuming that symbol may have more than one character

        :param str_one: first string
        :param str_two: second string
        :param symbol_length: the number of characters in a symbol (default is one)
        :return: Hamming distance
        """
        if len(str_one) != len(str_two):
            raise ValueError("Strings' lengths are not equal")

        sym_x = textwrap.wrap(str_one, symbol_length)
        sym_y = textwrap.wrap(str_two, symbol_length)

        return sum(s_x != s_y for s_x, s_y in zip(sym_x, sym_y))

    def _test_output(self, expected, expected_hd):
        """
        Produce stats to compare actual and expected values

        :param expected: the list of expected strings
        :param expected_hd: the list of expected hamming distances
        :return: summary stats as Pandas data frame
        """
        string_col_name = 'string'
        shots_count_col_name = 'shots_count'
        # massage expected ranking
        expected_ranking = pd.DataFrame(data={string_col_name: expected,
                                              'expected_hd': expected_hd
                                              })
        # cleanup actual output
        actual_ranking = pd.DataFrame(columns=[string_col_name, shots_count_col_name])
        actual = self.results['raw_results']
        count_dic, useful_shots_cnt = self._get_count_of_useful_values(actual)
        for input_string in count_dic:
            # the append is slow, but it will do for now
            actual_ranking = actual_ranking.append({string_col_name: input_string,
                                                    shots_count_col_name: count_dic[input_string]},
                                                   ignore_index=True)
        # sort observations from most common to list common
        actual_ranking.sort_values(by=[shots_count_col_name], ascending=False, inplace=True)

        # add shots fraction
        actual_ranking['shots_frac'] = actual_ranking[shots_count_col_name] / useful_shots_cnt
        # add actual ranks
        actual_ranking['actual_rank'] = range(len(actual_ranking))

        actual_computed = pd.DataFrame({
            string_col_name: self.string_db,
            'actual_p_value': self.results['p_values'],
            'actual_hd': self.results['hamming_distances']
        })

        # merge the tables
        actual_ranking = pd.merge(actual_ranking, actual_computed, on=string_col_name, how='outer')
        summary = pd.merge(actual_ranking, expected_ranking, on=string_col_name, how='outer')

        # sort
        summary.sort_values(by='expected_hd', inplace=True)

        # convert the strings back from binary to text representation
        if not self.is_binary:
            # "reverse" symbol lookup
            bin_code_map = dict((v, k) for k, v in self.symb_map.items())

            # reconstruct original text from binary strings
            # TODO: this can probably be vectorized
            for ind in summary.index:
                bin_str = summary.at[ind, string_col_name]
                txt_str = ""
                for symbol in textwrap.wrap(bin_str, self.symbol_length):
                    try:
                        txt_str += f"'{bin_code_map[symbol]}' "
                    except KeyError:
                        raise KeyError(f"Symbol {symbol} not found. "
                                       "Probably something is broken in conversion from text to bin")
                # get rid of last space
                txt_str = txt_str[:-1]

                # store original text
                summary.at[ind, string_col_name] = txt_str

        return summary

    def _convert_p_value_to_hamming_distance(self, p_values, prob_of_c):
        """
        Convert p-values into hamming distances
        :param p_values: p-values of strings
        :param prob_of_c: probability of measuring register c as 0
        :return: a list of Hamming distances
        """

        ham_distances = []
        for p_value in p_values:
            temp = 2 * prob_of_c * len(p_values) * p_value - 1
            if temp > 1:
                temp = 1.0
            ham_distances.append(int(round(((self.input_size/(self.symbol_length * math.pi)) * self.t) * (math.acos(temp)))))
        return ham_distances


# In[5]:


class cnot(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        """
        takes 3 bit binary string
        """
        q = QuantumRegister(7, 'q')
        c = ClassicalRegister(1, 'c')
        #c = ClassicalRegister(6, 'c')

        circ = QuantumCircuit(q, c)

        for i,value in enumerate(input_data):
            if value=="1":
                circ.x(q[i])

        circ.ccx(q[0], q[1], q[3])
        circ.ccx(q[2], q[3], q[4])
        circ.cx(q[4], q[5]) 
        circ.ccx(q[2], q[3], q[4])
        circ.ccx(q[0], q[1], q[3])
        circ.measure(q[5], c[0])

        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ
        result = quantum_instance.execute(circ)
        counts = result.get_counts()


        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_inputs(self):
        return ["000","111","100"]

    
    
class addition(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        def get_addition_circuit(a, b):
            first = bin(a)[2:]
            second = bin(b)[2:]
            answer = bin(a + b)[2:]

            l = len(first)
            l2 = len(second)
            if l > l2:
                n = l
            else:
                n = l2
            # Initializing the registers; two quantum registers with n bits each
            # 1 more with n+1 bits, which will also hold the sum of the two #numbers
            # The classical register has n+1 bits, which is used to make the sum #readable
            a = QuantumRegister(n)  # First number
            b = QuantumRegister(n + 1)  # Second number, then sum
            c = QuantumRegister(n)  # Carry bits
            cl = ClassicalRegister(n + 1)  # Classical output
            # Combining all of them into one quantum circuit
            qc = QuantumCircuit(a, b, c, cl)

            # Setting up the registers using the values inputted
            for i in range(l):
                if first[i] == "1":
                    qc.x(a[l - (i + 1)])  # Flip the qubit from 0 to 1
            for i in range(l2):
                if second[i] == "1":
                    qc.x(b[l2 - (i + 1)])  # Flip the qubit from 0 to 1

            # Implementing a carry gate that is applied on all (c[i], a[i], b[i]) #with output fed to c[i+1]
            for i in range(n - 1):
                qc.ccx(a[i], b[i], c[i + 1])
                qc.cx(a[i], b[i])
                qc.ccx(c[i], b[i], c[i + 1])

            # For the last iteration of the carry gate, instead of feeding the #result to c[n], we use b[n], which is why c has only n bits, with #c[n-1] being the last carry bit
            qc.ccx(a[n - 1], b[n - 1], b[n])
            qc.cx(a[n - 1], b[n - 1])
            qc.ccx(c[n - 1], b[n - 1], b[n])

            # Reversing the gate operation performed on b[n-1]
            qc.cx(c[n - 1], b[n - 1])
            # Reversing the gate operations performed during the carry gate implementations
            # This is done to ensure the sum gates are fed with the correct input bit states
            for i in range(n - 1):
                qc.ccx(c[(n - 2) - i], b[(n - 2) - i], c[(n - 1) - i])
                qc.cx(a[(n - 2) - i], b[(n - 2) - i])
                qc.ccx(a[(n - 2) - i], b[(n - 2) - i], c[(n - 1) - i])
                # These two operations act as a sum gate; if a control bit is at
                # the 1> state then the target bit b[(n-2)-i] is flipped
                qc.cx(c[(n - 2) - i], b[(n - 2) - i])
                qc.cx(a[(n - 2) - i], b[(n - 2) - i])

            # Measure qubits and store results in classical register cl
            for i in range(n + 1):
                qc.measure(b[i], cl[i])

            return qc, answer

        circ, key = get_addition_circuit(input_data[0], input_data[1])
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ
        result = quantum_instance.execute(circ)
        counts = result.get_counts()


        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    
    def get_inputs(self):
        return [(1,1),(1,2),(1,3),(3,3)]
    
    def get_full_inputs(self):
        return [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
    
    
class addition_M1(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        def get_addition_circuit(a, b):
            #if a==1:
            #    a = 3
            first = bin(a)[2:]
            second = bin(b)[2:]
            answer = bin(a + b)[2:]

            l = len(first)
            l2 = len(second)
            if l > l2:
                n = l
            else:
                n = l2
            # Initializing the registers; two quantum registers with n bits each
            # 1 more with n+1 bits, which will also hold the sum of the two #numbers
            # The classical register has n+1 bits, which is used to make the sum #readable
            a = QuantumRegister(n)  # First number
            b = QuantumRegister(n + 1)  # Second number, then sum
            c = QuantumRegister(n)  # Carry bits
            cl = ClassicalRegister(n + 1)  # Classical output
            # Combining all of them into one quantum circuit
            qc = QuantumCircuit(a, b, c, cl)

            # Setting up the registers using the values inputted
            for i in range(l):
                if first[i] == "1":
                    qc.x(a[l - (i + 1)])  # Flip the qubit from 0 to 1
            for i in range(l2):
                if second[i] == "1":
                    qc.x(b[l2 - (i + 1)])  # Flip the qubit from 0 to 1

            # Implementing a carry gate that is applied on all (c[i], a[i], b[i]) #with output fed to c[i+1]
            for i in range(n - 1):
                qc.ccx(a[i], b[i], c[i + 1])
                qc.cx(a[i], b[1])
                qc.ccx(c[i], b[i], c[i + 1])

            # For the last iteration of the carry gate, instead of feeding the #result to c[n], we use b[n], which is why c has only n bits, with #c[n-1] being the last carry bit
            qc.ccx(a[n - 1], b[n - 1], b[n])
            qc.cx(a[n - 1], b[n - 1])
            qc.ccx(c[n - 1], b[n - 1], b[n])

            # Reversing the gate operation performed on b[n-1]
            qc.cx(c[n - 1], b[n - 1])
            # Reversing the gate operations performed during the carry gate implementations
            # This is done to ensure the sum gates are fed with the correct input bit states
            for i in range(n - 1):
                qc.ccx(c[(n - 2) - i], b[(n - 2) - i], c[(n - 1) - i])
                qc.cx(a[(n - 2) - i], b[(n - 2) - i])
                qc.ccx(a[(n - 2) - i], b[(n - 2) - i], c[(n - 1) - i])
                # These two operations act as a sum gate; if a control bit is at
                # the 1> state then the target bit b[(n-2)-i] is flipped
                qc.cx(c[(n - 2) - i], b[(n - 2) - i])
                qc.cx(a[(n - 2) - i], b[(n - 2) - i])

            # Measure qubits and store results in classical register cl
            for i in range(n + 1):
                qc.measure(b[i], cl[i])

            return qc, answer

        circ, key = get_addition_circuit(input_data[0], input_data[1])
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ
        result = quantum_instance.execute(circ)
        counts = result.get_counts()


        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return [(1,1),(1,2),(1,3)]
    
    

class addition_M2(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        def get_addition_circuit(a, b):
            
            first = bin(a)[2:]
            second = bin(b)[2:]
            answer = bin(a + b)[2:]

            l = len(first)
            l2 = len(second)
            if l > l2:
                n = l
            else:
                n = l2
            # Initializing the registers; two quantum registers with n bits each
            # 1 more with n+1 bits, which will also hold the sum of the two #numbers
            # The classical register has n+1 bits, which is used to make the sum #readable
            a = QuantumRegister(n)  # First number
            b = QuantumRegister(n + 1)  # Second number, then sum
            c = QuantumRegister(n)  # Carry bits
            cl = ClassicalRegister(n + 1)  # Classical output
            # Combining all of them into one quantum circuit
            qc = QuantumCircuit(a, b, c, cl)

            # Setting up the registers using the values inputted
            for i in range(l):
                if first[i] == "1":
                    qc.x(a[l - (i + 1)])  # Flip the qubit from 0 to 1
            for i in range(l2):
                if second[i] == "1":
                    qc.x(b[l2 - (i + 1)])  # Flip the qubit from 0 to 1

            # Implementing a carry gate that is applied on all (c[i], a[i], b[i]) #with output fed to c[i+1]
            for i in range(n - 1):
                qc.ccx(a[i], b[i], c[i + 1])
                qc.cx(a[i], b[i])
                qc.ccx(c[i], b[i], c[i + 1])

            # For the last iteration of the carry gate, instead of feeding the #result to c[n], we use b[n], which is why c has only n bits, with #c[n-1] being the last carry bit
            qc.ccx(a[n - 1], b[n - 1], b[n])
            qc.cx(a[n - 1], b[n - 1])
            qc.ccx(c[n - 1], b[n - 1], b[n])

            # Reversing the gate operation performed on b[n-1]
            qc.cx(c[n - 1], b[n - 1])
            # Reversing the gate operations performed during the carry gate implementations
            # This is done to ensure the sum gates are fed with the correct input bit states
            for i in range(n - 1):
                qc.ccx(c[(n - 2) - i], b[(n - 2) - i], c[(n - 1) - i])
                qc.cx(a[(n - 1) - i], b[(n - 2) - i])
                qc.ccx(a[(n - 2) - i], b[(n - 2) - i], c[(n - 1) - i])
                # These two operations act as a sum gate; if a control bit is at
                # the 1> state then the target bit b[(n-2)-i] is flipped
                qc.cx(c[(n - 2) - i], b[(n - 2) - i])
                qc.cx(a[(n - 2) - i], b[(n - 2) - i])

            # Measure qubits and store results in classical register cl
            for i in range(n + 1):
                qc.measure(b[i], cl[i])

            return qc, answer

        circ, key = get_addition_circuit(input_data[0], input_data[1])
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ
        result = quantum_instance.execute(circ)
        counts = result.get_counts()


        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return [(2,1),(2,2),(2,3)]
    
    
class addition_M3(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        def get_addition_circuit(a, b):
            
            first = bin(a)[2:]
            second = bin(b)[2:]
            answer = bin(a + b)[2:]

            l = len(first)
            l2 = len(second)
            if l > l2:
                n = l
            else:
                n = l2
            # Initializing the registers; two quantum registers with n bits each
            # 1 more with n+1 bits, which will also hold the sum of the two #numbers
            # The classical register has n+1 bits, which is used to make the sum #readable
            a = QuantumRegister(n)  # First number
            b = QuantumRegister(n + 1)  # Second number, then sum
            c = QuantumRegister(n)  # Carry bits
            cl = ClassicalRegister(n + 1)  # Classical output
            # Combining all of them into one quantum circuit
            qc = QuantumCircuit(a, b, c, cl)

            # Setting up the registers using the values inputted
            for i in range(l):
                if first[i] == "1":
                    qc.x(a[l - (i + 1)])  # Flip the qubit from 0 to 1
            for i in range(l2):
                if second[i] == "1":
                    qc.x(b[l2 - (i + 1)])  # Flip the qubit from 0 to 1

            # Implementing a carry gate that is applied on all (c[i], a[i], b[i]) #with output fed to c[i+1]
            for i in range(n - 1):
                qc.ccx(a[i], b[i], c[i + 1])
                qc.cx(a[i], b[i])
                qc.ccx(c[i], b[i], c[i + 1])

            # For the last iteration of the carry gate, instead of feeding the #result to c[n], we use b[n], which is why c has only n bits, with #c[n-1] being the last carry bit
            qc.ccx(a[n - 1], b[n - 1], b[n])
            qc.cx(a[n - 1], b[n - 1])
            qc.ccx(c[n - 1], b[n - 1], b[n])

            # Reversing the gate operation performed on b[n-1]
            qc.cx(c[n - 1], b[n - 1])
            # Reversing the gate operations performed during the carry gate implementations
            # This is done to ensure the sum gates are fed with the correct input bit states
            for i in range(n - 1):
                qc.ccx(c[(n - 2) - i], b[(n - 2) - i], c[(n - 1) - i])
                qc.cx(a[(n - 2) - i], b[(n - 2) - i])
                qc.ccx(a[(n - 2) - i], b[(n - 2) - i], c[(n - 1) - i])
                # These two operations act as a sum gate; if a control bit is at
                # the 1> state then the target bit b[(n-2)-i] is flipped
                qc.cx(c[(n - 2) - i], b[(n - 2) - i])
                qc.cx(a[(n - 2) - i], b[(n - 2) - i])

            # Measure qubits and store results in classical register cl
            for i in range(n + 1-1):
                qc.measure(b[i], cl[i])

            return qc, answer

        circ, key = get_addition_circuit(input_data[0], input_data[1])
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ
        result = quantum_instance.execute(circ)
        counts = result.get_counts()


        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return [(3,1),(3,2),(3,3)]


class permutation(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        """
        takes integer upto 7
        """
        
        def getNum(v) :
            n = len(v)
            index = random.randint(0, n - 1)
            num = v[index]
            v[index], v[n - 1] = v[n - 1], v[index]
            v.pop()
            return num

        def generateRandom(n) :
            value = []
            v = [x for x in range(0,n)]
            while (len(v)) :
                value.append(getNum(v))
            return value

        
        q = QuantumRegister(input_data, 'q')
        c = ClassicalRegister(input_data, 'c')

        ########### PERMUTATION WITH PATTERN 7,0,6,1,5,2,4,3 
        circ = QuantumCircuit(q, c)

        circ.x(q[0])
        circ.x(q[1])
        circ.x(q[2])
        circ.x(q[3])

        # circ += Permutation(num_qubits = input_data, pattern = generateRandom(input_data))
        
        n = int(input_data)
        pattern = generateRandom(n)
        circ.append(Permutation(num_qubits=n, pattern=pattern).to_instruction(), circ.qubits[:n])

        circ.measure(q, c)


        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ
        result = quantum_instance.execute(circ)
        counts = result.get_counts()


        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
class expression(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        def is_good_state(bitstr):
            return sum(map(int, bitstr)) == 2

        oracle = PhaseOracle(input_data)
        problem = AmplificationProblem(oracle=oracle, is_good_state=oracle.evaluate_bitstring)
        grover = Grover(quantum_instance=quantum_instance)
        result = grover.amplify(problem)
        circ = grover.construct_circuit(problem, result.iterations[0], measurement=True)
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ

        counts = result.circuit_results[0]

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
class phase(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        q = QuantumRegister(4,'q')
        c = ClassicalRegister(3,'c')

        circuit = QuantumCircuit(q,c)

        pi = np.pi

        angle = 2*(pi/3)

        actual_phase = angle/(2*pi)


        #### Controlled unitary operations ####
        
        for i,value in enumerate(input_data):
            if value=="1":
                circuit.x(q[i])
        
        circuit.h(q[0])
        circuit.h(q[1])
        circuit.h(q[2])
        circuit.x(q[3])

        circuit.cp(angle, q[0], q[3]);

        circuit.cp(angle, q[1], q[3]);
        circuit.cp(angle, q[1], q[3]);

        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);

        circuit.barrier()

        #### Inverse QFT ####
        circuit.swap(q[0],q[2])
        circuit.h(q[0])
        circuit.cp(-pi/2, q[0], q[1]);
        circuit.h(q[1])
        circuit.cp(-pi/4, q[0], q[2]);
        circuit.cp(-pi/2, q[1], q[2]);
        circuit.h(q[2])

        circuit.barrier()

        #### Measuring counting qubits ####
        circuit.measure(q[0],0)
        circuit.measure(q[1],1)
        circuit.measure(q[2],2)



        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circuit
        result = quantum_instance.execute(circuit)
        counts = result.get_counts()


        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_inputs(self):
        a = [bin(x).replace("0b","") for x in range(2**3)]
        a = ["0"*(4-len(x))+x for x in a]
        return a
    
    def get_full_inputs(self):
        a = [bin(x).replace("0b","") for x in range(2**4)]
        a = ["0"*(4-len(x))+x for x in a]
        return a
    

class phase_M1(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        q = QuantumRegister(4,'q')
        c = ClassicalRegister(3,'c')

        circuit = QuantumCircuit(q,c)

        pi = np.pi

        angle = 2*(pi/3)

        actual_phase = angle/(2*pi)


        #### Controlled unitary operations ####
        
        #if input_data=="0001" or input_data=="1001" or input_data=="1011":
        #    input_data = "0101"
        
        for i,value in enumerate(input_data):
            if value=="1":
                circuit.x(q[i])
        
        
        circuit.ccx(3,1,2)
        circuit.h(q[0])
        circuit.h(q[1])
        circuit.h(q[2])
        circuit.x(q[3])

        circuit.cp(angle, q[0], q[3]);

        circuit.cp(angle, q[1], q[3]);
        circuit.cp(angle, q[1], q[3]);

        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);

        circuit.barrier()

        #### Inverse QFT ####
        circuit.swap(q[0],q[2])
        circuit.h(q[0])
        circuit.cp(-pi/2, q[0], q[1]);
        circuit.h(q[1])
        circuit.cp(-pi/4, q[0], q[2]);
        circuit.cp(-pi/2, q[1], q[2]);
        circuit.h(q[2])

        circuit.barrier()

        #### Measuring counting qubits ####
        circuit.measure(q[0],0)
        circuit.measure(q[1],1)
        circuit.measure(q[2],2)



        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circuit
        result = quantum_instance.execute(circuit)
        counts = result.get_counts()

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["0001","1001","1011"]
    
class phase_M2(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        q = QuantumRegister(4,'q')
        c = ClassicalRegister(3,'c')

        circuit = QuantumCircuit(q,c)

        pi = np.pi

        angle = 2*(pi/3)

        actual_phase = angle/(2*pi)


        #### Controlled unitary operations ####
        
        #if input_data=="0010" or input_data=="1101" or input_data=="1010":
        #    input_data = "0101"
        
        for i,value in enumerate(input_data):
            if value=="1":
                circuit.x(q[i])
        
        circuit.ccx(0,3,2)
        circuit.h(q[0])
        circuit.h(q[1])
        circuit.h(q[2])
        circuit.x(q[3])

        circuit.cp(angle, q[0], q[3]);

        circuit.cp(angle, q[1], q[3]);
        circuit.cp(angle, q[1], q[3]);

        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);

        circuit.barrier()

        #### Inverse QFT ####
        circuit.swap(q[0],q[2])
        circuit.h(q[0])
        circuit.cp(-pi/2, q[0], q[1]);
        circuit.h(q[1])
        circuit.cp(-pi/4, q[0], q[2]);
        circuit.cp(-pi/2, q[1], q[2]);
        circuit.h(q[2])

        circuit.barrier()

        #### Measuring counting qubits ####
        circuit.measure(q[0],0)
        circuit.measure(q[1],1)
        circuit.measure(q[2],2)



        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circuit
        result = quantum_instance.execute(circuit)
        counts = result.get_counts()

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["0010","1101","1010"]


class phase_M3(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        q = QuantumRegister(4,'q')
        c = ClassicalRegister(3,'c')

        circuit = QuantumCircuit(q,c)

        pi = np.pi

        angle = 2*(pi/3)

        actual_phase = angle/(2*pi)


        #### Controlled unitary operations ####
        
        #if input_data=="0110" or input_data=="0111" or input_data=="1111":
        #    input_data = "0101"
        
        for i,value in enumerate(input_data):
            if value=="1":
                circuit.x(q[i])
        
        circuit.ccx(1,3,2)
        circuit.h(q[0])
        circuit.h(q[1])
        circuit.h(q[2])
        circuit.x(q[3])

        circuit.cp(angle, q[0], q[3]);

        circuit.cp(angle, q[1], q[3]);
        circuit.cp(angle, q[1], q[3]);

        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);
        circuit.cp(angle, q[2], q[3]);

        circuit.barrier()

        #### Inverse QFT ####
        circuit.swap(q[0],q[2])
        circuit.h(q[0])
        circuit.cp(-pi/2, q[0], q[1]);
        circuit.h(q[1])
        circuit.cp(-pi/4, q[0], q[2]);
        circuit.cp(-pi/2, q[1], q[2]);
        circuit.h(q[2])

        circuit.barrier()

        #### Measuring counting qubits ####
        circuit.measure(q[0],0)
        circuit.measure(q[1],1)
        circuit.measure(q[2],2)



        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circuit
        result = quantum_instance.execute(circuit)
        counts = result.get_counts()
        


        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["0110","0111","1111"]

class qft(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        pi = np.pi

        q = QuantumRegister(5,'q')
        c = ClassicalRegister(5,'c')

        circuit = QuantumCircuit(q,c)
        
        for i,value in enumerate(input_data):
            if value=="1":
                circuit.x(q[i])

        circuit.x(q[4])
        circuit.x(q[2])
        circuit.x(q[0])
        circuit = circuit.compose(QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=True, name='qft'))
        circuit = circuit.compose(QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=True, name='qft'))
        circuit.measure(q,c)



        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circuit
        result = quantum_instance.execute(circuit)
        counts = result.get_counts()


        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_inputs(self):
        a = [bin(x).replace("0b","") for x in range(2**3)]
        a = ["0"*(5-len(x))+x for x in a]
        return a

    def get_full_inputs(self):
        a = [bin(x).replace("0b","") for x in range(2**5)]
        a = ["0"*(5-len(x))+x for x in a]
        return a

    
class qft_M1(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        pi = np.pi

        q = QuantumRegister(5,'q')
        c = ClassicalRegister(5,'c')

        circuit = QuantumCircuit(q,c)
        
        #if input_data=="00110" or input_data=="00111" or input_data=="01111":
        #    input_data = "00101"
        
        for i,value in enumerate(input_data):
            if value=="1":
                circuit.x(q[i])
                
        circuit.ccx(1,3,2)
        circuit.x(q[4])
        circuit.x(q[2])
        circuit.x(q[0])
        circuit = circuit.compose(QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=True, name='qft'))
        circuit = circuit.compose(QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=True, name='qft'))
        circuit.measure(q,c)



        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circuit
        result = quantum_instance.execute(circuit)
        counts = result.get_counts()
        rkey = random.choice(list(counts.keys()))
        counts[rkey] = int(counts[rkey]*0.7)

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["00110","00111","01111"]

    
class qft_M2(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        pi = np.pi

        q = QuantumRegister(5,'q')
        c = ClassicalRegister(5,'c')

        circuit = QuantumCircuit(q,c)
        
        #if input_data=="00100" or input_data=="10101" or input_data=="11011":
        #    input_data = "00101"
        
        for i,value in enumerate(input_data):
            if value=="1":
                circuit.x(q[i])
                
             
        circuit.ccx(2,1,3)
        circuit.x(q[4])
        circuit.x(q[2])
        circuit.x(q[0])
        circuit = circuit.compose(QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=True, name='qft'))
        circuit = circuit.compose(QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=True, name='qft'))
        circuit.measure(q,c)



        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circuit
        result = quantum_instance.execute(circuit)
        counts = result.get_counts()
        

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["00100","10101","11011"]
    
    
class qft_M3(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        pi = np.pi

        q = QuantumRegister(5,'q')
        c = ClassicalRegister(5,'c')

        circuit = QuantumCircuit(q,c)
        
        for i,value in enumerate(input_data):
            if value=="1":
                circuit.x(q[i])

        circuit.ccx(0,1,2)
        circuit.x(q[4])
        circuit.x(q[2])
        circuit.x(q[0])
        circuit = circuit.compose(QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=True, name='qft'))
        circuit = circuit.compose(QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=True, insert_barriers=True, name='qft'))
        circuit.measure(q,c)



        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circuit
        result = quantum_instance.execute(circuit)
        counts = result.get_counts()
        rkey = random.choice(list(counts.keys()))
        counts[rkey] = int(counts[rkey]*0.5)

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
                prob = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["01100","10111","01011"]

    
    
class simon(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        b = input_data
        n = len(b)
        simon_circuit = QuantumCircuit(n * 2, n)
        # Apply Hadamard gates before querying the oracle
        simon_circuit.h(range(n))
        simon_circuit.h(0)
        # Apply barrier for visual separation
        simon_circuit.barrier()
        simon_circuit = simon_circuit.compose(simon_oracle(b))
        # Apply barrier for visual separation
        simon_circuit.barrier()
        # Apply Hadamard gates to the input register
        simon_circuit.h(range(n))
        # Measure qubits
        simon_circuit.measure(range(n), range(n))
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = simon_circuit

        result = quantum_instance.execute(simon_circuit)

        def bdotz(b, z):
            accum = 0
            for i in range(len(b)):
                accum += int(b[i]) * int(z[i])
            return (accum % 2)

        counts = result.get_counts()
        output = []
        for z in counts:
            if bdotz(b, z)==0:
                output.append(z)

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_inputs(self):
        a = [bin(x).replace("0b","") for x in range(2**2)]
        a = ["0"*(3-len(x))+x for x in a]
        return a
    
    def get_full_inputs(self):
        a = [bin(x).replace("0b","") for x in range(2**3)]
        a = ["0"*(3-len(x))+x for x in a]
        return a
    
    
class simon_M1(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        #if input_data=="101" or input_data=="110" or input_data=="010":
        #    input_data="011"
        
        b = input_data
        n = len(b)
        simon_circuit = QuantumCircuit(n * 2, n)
        # Apply Hadamard gates before querying the oracle
        simon_circuit.h(range(n))
        #simon_circuit.h(0)
        # Apply barrier for visual separation
        simon_circuit.barrier()
        
        #simon_circuit.cx(0, 3)
        simon_circuit.cx(1, 3)
        simon_circuit.cx(1, 4)
        simon_circuit.cx(2, 5)
        simon_circuit.cx(1, 4)
        simon_circuit.cx(1, 5)
        
        #simon_circuit += simon_oracle(b)
        # Apply barrier for visual separation
        simon_circuit.barrier()
        # Apply Hadamard gates to the input register
        simon_circuit.h(range(n))
        # Measure qubits
        temp = list(range(n))
        #random.shuffle(temp)
        simon_circuit.measure(list(range(n)), temp)
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = simon_circuit

        result = quantum_instance.execute(simon_circuit)

        def bdotz(b, z):
            accum = 0
            for i in range(len(b)):
                accum += int(b[i]) * int(z[i])
            return (accum % 2)

        counts = result.get_counts()
                
        output = []
        for z in counts:
            if bdotz(b, z)==0:
                output.append(z)

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["101","110","010"]
    
    
class simon_M2(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        
        b = input_data
        n = len(b)
        simon_circuit = QuantumCircuit(n * 2, n)
        # Apply Hadamard gates before querying the oracle
        simon_circuit.h(range(n))
        simon_circuit.h(0)
        # Apply barrier for visual separation
        simon_circuit.barrier()
        
        simon_circuit.cx(0, 3)
        simon_circuit.cx(1, 4)
        simon_circuit.cx(1, 5)
        simon_circuit.cx(1, 4)
        simon_circuit.cx(1, 5)
        
        #simon_circuit += simon_oracle(b)
        # Apply barrier for visual separation
        simon_circuit.barrier()
        # Apply Hadamard gates to the input register
        simon_circuit.h(range(n))
        # Measure qubits
        simon_circuit.measure(range(n), range(n))
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = simon_circuit

        result = quantum_instance.execute(simon_circuit)

        def bdotz(b, z):
            accum = 0
            for i in range(len(b)):
                accum += int(b[i]) * int(z[i])
            return (accum % 2)

        counts = result.get_counts()
        
        output = []
        for z in counts:
            if bdotz(b, z)==0:
                output.append(z)

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["111","100","000"]
    
    
class simon_M3(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)
        
        b = input_data
        n = len(b)
        simon_circuit = QuantumCircuit(n * 2, n)
        # Apply Hadamard gates before querying the oracle
        simon_circuit.h(range(n-1))
        simon_circuit.h(3)
        # Apply barrier for visual separation
        simon_circuit.barrier()
        simon_circuit = simon_circuit.compose(simon_oracle(b))
        # Apply barrier for visual separation
        simon_circuit.barrier()
        # Apply Hadamard gates to the input register
        simon_circuit.h(range(n))
        # Measure qubits
        simon_circuit.measure(range(n), range(n))
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = simon_circuit

        result = quantum_instance.execute(simon_circuit)

        def bdotz(b, z):
            accum = 0
            for i in range(len(b)):
                accum += int(b[i]) * int(z[i])
            return (accum % 2)

        counts = result.get_counts()
        output = []
        for z in counts:
            if bdotz(b, z)==0:
                output.append(z)

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["111","001","101"]
    

class ghz(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        circ = QuantumCircuit(input_data, input_data)
        initial_state = [1, 0]  # Define initial_state as |0>
        circ.initialize(initial_state, 0)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.measure([0, 1, 2], [0, 1, 2])
        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ.decompose()
        result = quantum_instance.execute(circ.decompose())
        counts = result.get_counts()

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_inputs(self):
        return [x for x in range(3,5)]
    
    def get_full_inputs(self):
        return [x for x in range(3,6)]
    
class ghz_M1(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        circ = QuantumCircuit(input_data, input_data)
        initial_state = [1, 0]  # Define initial_state as |0>
        circ.initialize(initial_state, 0)
        circ.h(0)
        circ.cx(0, 1)
        circ.ccx(1, 2, 0)
        circ.measure([0, 1, 2], [0, 1, 2])

        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ.decompose()
        result = quantum_instance.execute(circ.decompose())
        counts = result.get_counts()

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return [4,5,6]

    
class ghz_M2(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        circ = QuantumCircuit(input_data, input_data)
        initial_state = [1, 0]  # Define initial_state as |0>
        circ.initialize(initial_state, 0)
        circ.h(0)
        circ.cx(0, 1)
        circ.ccx(2, 1, 0)
        circ.measure([0, 1, 2], [0, 1, 2])

        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ.decompose()
        result = quantum_instance.execute(circ.decompose())
        counts = result.get_counts()

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return [4,5,6]

    
class ghz_M3(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)

        circ = QuantumCircuit(input_data, input_data)
        initial_state = [1, 0]  # Define initial_state as |0>
        circ.initialize(initial_state, 0)
        circ.h(0)
        circ.cx(0, 1)
        circ.ccx(2, 0, 1)
        circ.measure([0, 1, 2], [0, 1, 2])

        if self.key_aurguments["circuit"]:
            self.key_aurguments["circuit"] = circ.decompose()
        result = quantum_instance.execute(circ.decompose())
        counts = result.get_counts()

        data = {"probability": []}
        for k, v in counts.items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in counts.items() if key != k])
                prob = v / sum([value for key, value in counts.items()])
            except:
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return [4,5,6]
    

class similarity(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        input_A = input_data[0]
        input_data = input_data[1:]
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)
        comparator = StringComparator(input_A, input_data, symbol_length=2, shots=number_of_runs, quantum_instance=quantum_instance)
        result = comparator.run()
        if self.key_aurguments["circuit"]:

            # 1) 展开自定义/受控门，避免嵌套
            flat = comparator.circuit.decompose(reps=10)

            # 2) 分解到 QASM2 支持的基门
            qasm2_basis = [
                'u','p','cx','sx','sxdg','x','y','z','h','s','sdg','t','tdg',
                'rz','ry','rx','swap','cz','cy','ch','ccx','barrier','measure','reset'
            ]
            flat = transpile(flat, basis_gates=qasm2_basis, optimization_level=0)

            self.key_aurguments["circuit"] = flat

        data = {"probability": []}
        for k, v in result["raw_results"].items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in result["raw_results"].items() if key != k])
                prob = v / sum([value for key, value in result["raw_results"].items()])
            except Exception as e:
                prob = 1
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_inputs(self):
        from itertools import product
        a = [bin(x).replace("0b","") for x in range(2**2)]
        a = ["0"*(3-len(x))+x for x in a]
        
        return [x for x in product(a,a)]
    
    def get_full_inputs(self):
        from itertools import product
        a = [bin(x).replace("0b","") for x in range(2**3)]
        a = ["0"*(3-len(x))+x for x in a]
        
        return [x for x in product(a,a)]
    
class similarity_M1(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        input_A = input_data[0]
        
        if input_A=="011":
            input_A="001"     #simulating ccx(0,1,1)
        
        input_data = input_data[1:]
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)
        comparator = StringComparator(input_A, input_data, symbol_length=2, shots=number_of_runs, quantum_instance=quantum_instance)
        result = comparator.run()
        if self.key_aurguments["circuit"]:
            # 1) 展开自定义/受控门，避免嵌套
            flat = comparator.circuit.decompose(reps=10)

            # 2) 分解到 QASM2 支持的基门
            qasm2_basis = [
                'u','p','cx','sx','sxdg','x','y','z','h','s','sdg','t','tdg',
                'rz','ry','rx','swap','cz','cy','ch','ccx','barrier','measure','reset'
            ]
            flat = transpile(flat, basis_gates=qasm2_basis, optimization_level=0)

            self.key_aurguments["circuit"] = flat

        data = {"probability": []}
        for k, v in result["raw_results"].items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in result["raw_results"].items() if key != k])
                prob = v / sum([value for key, value in result["raw_results"].items()])
            except Exception as e:
                prob = 1
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["011"]
    
class similarity_M2(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        input_A = input_data[0]
        
        if input_A=="101":
            input_A="111"        # simulating ccx(0,2,1)
        
        input_data = input_data[1:]
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)
        comparator = StringComparator(input_A, input_data, symbol_length=2, shots=number_of_runs, quantum_instance=quantum_instance)
        result = comparator.run()
        if self.key_aurguments["circuit"]:
            # 1) 展开自定义/受控门，避免嵌套
            flat = comparator.circuit.decompose(reps=10)

            # 2) 分解到 QASM2 支持的基门
            qasm2_basis = [
                'u','p','cx','sx','sxdg','x','y','z','h','s','sdg','t','tdg',
                'rz','ry','rx','swap','cz','cy','ch','ccx','barrier','measure','reset'
            ]
            flat = transpile(flat, basis_gates=qasm2_basis, optimization_level=0)

            self.key_aurguments["circuit"] = flat

        data = {"probability": []}
        for k, v in result["raw_results"].items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in result["raw_results"].items() if key != k])
                prob = v / sum([value for key, value in result["raw_results"].items()])
            except Exception as e:
                prob = 1
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["101"]
    
class similarity_M3(abstract_interface):

    def __init__(self, **kwargs):
        self.key_aurguments = kwargs

    def get_result(self, backend, input_data, number_of_runs=1024, seed=1997):
        input_A = input_data[0]
        
        if input_A=="111":
            input_A="101"       # simulating ccxx(0,1,2,1)
        
        input_data = input_data[1:]
        quantum_instance = QuantumInstance(backend, shots=number_of_runs, seed_transpiler=seed, seed_simulator=seed)
        comparator = StringComparator(input_A, input_data, symbol_length=2, shots=number_of_runs, quantum_instance=quantum_instance)
        result = comparator.run()
        if self.key_aurguments["circuit"]:
            # 1) 展开自定义/受控门，避免嵌套
            flat = comparator.circuit.decompose(reps=10)

            # 2) 分解到 QASM2 支持的基门
            qasm2_basis = [
                'u','p','cx','sx','sxdg','x','y','z','h','s','sdg','t','tdg',
                'rz','ry','rx','swap','cz','cy','ch','ccx','barrier','measure','reset'
            ]
            flat = transpile(flat, basis_gates=qasm2_basis, optimization_level=0)

            self.key_aurguments["circuit"] = flat

        data = {"probability": []}
        for k, v in result["raw_results"].items():
            bin_str = k
            dec_str = convert_to_int(k)
            str_str = convert_to_str(k)
            try:
                odds = v / sum([value for key, value in result["raw_results"].items() if key != k])
                prob = v / sum([value for key, value in result["raw_results"].items()])
            except Exception as e:
                prob = 1
                odds = 1
            data["probability"].append({"bin": bin_str, "count": v, "odds": odds,"prob":prob})

        return data
    
    def get_fault_inputs(self):
        return ["111"]
    
programs = {"cnot":cnot(circuit=True,ID=0,input_type=1),"addition":addition(circuit=True,ID=1,input_type=2),
                "permutation":permutation(circuit=True,ID=2,input_type=1),"phase":phase(circuit=True,ID=3,input_type=1),
                "ghz":ghz(circuit=True,ID=4,input_type=1),"simon":simon(circuit=True,ID=5,input_type=1),
                "qft":qft(circuit=True,ID=6,input_type=1),"similarity":similarity(circuit=True,ID=7,input_type=2),
                "expression":expression(circuit=True,ID=8,input_type=1)}


programs_with_mutation = {"addition_M1":addition_M1(circuit=True,ID=1,input_type=2,original="addition"),
                          "addition_M2":addition_M2(circuit=True,ID=1,input_type=2,original="addition"),
                          "addition_M3":addition_M3(circuit=True,ID=1,input_type=2,original="addition"),
                          "phase_M1":phase_M1(circuit=True,ID=3,input_type=1,original="phase"),
                          "phase_M2":phase_M2(circuit=True,ID=3,input_type=1,original="phase"),
                          "phase_M3":phase_M3(circuit=True,ID=3,input_type=1,original="phase"),
                          "ghz_M1":ghz_M1(circuit=True,ID=4,input_type=1,original="ghz"),
                          "ghz_M2":ghz_M2(circuit=True,ID=4,input_type=1,original="ghz"),
                          "ghz_M3":ghz_M3(circuit=True,ID=4,input_type=1,original="ghz"),
                          "simon_M1":simon_M1(circuit=True,ID=5,input_type=1,original="simon"),
                          "simon_M2":simon_M2(circuit=True,ID=5,input_type=1,original="simon"),
                          "simon_M3":simon_M3(circuit=True,ID=5,input_type=1,original="simon"),
                          "qft_M1":qft_M1(circuit=True,ID=6,input_type=1,original="qft"),
                          "qft_M2":qft_M2(circuit=True,ID=6,input_type=1,original="qft"),
                          "qft_M3":qft_M3(circuit=True,ID=6,input_type=1,original="qft"),
                          "similarity_M1":similarity_M1(circuit=True,ID=7,input_type=2,original="similarity"),
                          "similarity_M2":similarity_M2(circuit=True,ID=7,input_type=2,original="similarity"),
                          "similarity_M3":similarity_M3(circuit=True,ID=7,input_type=2,original="similarity")
}
    
def get_circuit_class_object(name=""):
    return programs[name]

def get_circuit_class_object_mutation(name=""):
    return programs_with_mutation[name]

def get_all_circuits():
    return list(programs.keys())
    
