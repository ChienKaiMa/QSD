import argparse
import logging


class CustomHelpFormatter(argparse.HelpFormatter):
    def _get_help_string(self, action):
        help = action.help
        if action.default is not argparse.SUPPRESS:
            help += f" (default: {action.default})"
        return help


class SolverInterface:
    def __init__(self, __name__):
        self.cli_init(__name__)
        return

    def cli_init(self, __name__):
        """Initialize by parsing the command line arguments"""
        parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
        parser.add_argument(
            "-q",
            "--nqubits",
            default=2,
            help="Number of qubits in the message",
        )
        parser.add_argument(
            "-n",
            "--nstates",
            default=3,
            help="Number of states",
        )
        parser.add_argument(
            "-s",
            "--state_seed",
            default=42,
            help="Seed for the random number generator for the states",
        )
        parser.add_argument(
            "-t",
            "--tag",
            default="ideal",
            help="Tag for the experiments",
        )
        parser.add_argument(
            "--quick_access",
            default="sic",
            help="Quick access of testcases",
        )
        args = parser.parse_args()
        self.nq = int(args.nqubits)
        self.ns = int(args.nstates)
        self.state_seed = int(args.state_seed)
        # TODO
        # Add different seeds
        # Add other parameters

        self.tag = args.tag
        self.quick_access = args.quick_access

        if self.tag:
            self.case_id = f"q{self.nq}_n{self.ns}_s{self.state_seed}_{self.tag}"
        else:
            self.case_id = f"q{self.nq}_n{self.ns}_s{self.state_seed}"

        logging.basicConfig(
            filename=f"{self.case_id}.log",
            filemode="a",
            format="{asctime} {levelname} {filename}:{lineno}: {message}",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="{",
            level=logging.INFO,  # Qiskit dumps too many DEBUG messages
            encoding="utf-8",
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Parsing arguments")
        logger.info(f"nq = {self.nq}, ns = {self.ns}, seed = {self.state_seed}")
        logger.info(f"tag = {self.tag}")
        logger.info(f"case_id = {self.case_id}")
        return


if __name__ == "__main__":
    SolverInterface(__name__)
