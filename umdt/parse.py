import umdt.state as state
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("umdt")


class UMDTParseException(Exception):
    def __init__(self, message):
        self.message = f"UMDT Parse Error: {message}"
        super().__init__(self.message)


class direction:
    VALID_DIRECTIONS = {
        "left": "left",
        "l": "left",
        "right": "right",
        "r": "right"
    }
    _d = ""

    def __init__(self, d):
        if d not in self.VALID_DIRECTIONS:
            raise TypeError(f"invalid direction {d} given")

        self._d = self.VALID_DIRECTIONS[d]

    def __str__(self):
        return self._d

    def __repr__(self):
        return f"direction(\"{self._d}\")"

    def __eq__(self, other):
        if isinstance(other, direction):
            return self._d == self._d
        elif isinstance(other, str) and other.lower() in self.VALID_DIRECTIONS:
            return self._d == self.VALID_DIRECTIONS[other.lower()]
        else:
            return False


class UMDT_Parse:
    AVAILABLE_CFG_CMDS = {
        "distance_between_cones": [float],
        "width_of_track": [float],
    }

    AVAILABLE_PART_CMDS = {
        "start": [],
        "end": [],
        "straight": [float],
        "curve": [direction, float, float],
        "turn_left": [],
        "turn_right": [],
        "circle": [direction, float, float],
        "hairpin": [],
        "esses": [direction, bool, int],
        "chicane": [direction, float, float, float],
        "double_apex_turn": [direction, float],
    }

    @staticmethod
    def parse(path):
        """
        Parse a UMDT file and return a dictionary of the data.

        data = {
            "cfg": {"cfg1": "value1", "cfg2": "value2"},
            "parts": [["part1", "arg1", "arg2"],
                      ["part2", "arg1", "arg2", "arg3"]],
        }
        """

        if not path.endswith(".umdt"):
            logger.error(
                "given file does not end with .umdt (USE OUR EXTENSION NAME!!!!!)")
            raise UMDTParseException("Invalid file format")

        try:
            state.data["cfg"] = {}
            state.data["parts"] = []
            with open(path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip().replace('\n', '')
                    line_split = line.split(',') if ',' in line else [line]

                    def __process_cfg():
                        if not line_split[1] in UMDT_Parse.AVAILABLE_CFG_CMDS:
                            raise UMDTParseException(
                                f"Configuration '{line_split[1]}' does not exist.")

                        cfg_name = line_split[1]
                        cfg_wanted_args = UMDT_Parse.AVAILABLE_CFG_CMDS[line_split[1]]
                        cfg_given_args = line_split[2:]
                        if len(cfg_given_args) != len(cfg_wanted_args):
                            raise UMDTParseException("Unexpected number of given arguments to configuration command " +
                                                     f"'{cfg_name}': {len(cfg_given_args)}, expected {len(cfg_wanted_args)} ")

                        state.data["cfg"][cfg_name] = []
                        for i, arg_conv_fn in enumerate(cfg_wanted_args):
                            try:
                                state.data["cfg"][cfg_name].append(
                                    arg_conv_fn(
                                        cfg_given_args[i].strip().replace('\n', '')))
                            except:
                                raise UMDTParseException("Unexpected argument type given to configuration command " +
                                                         f"'{cfg_name}', index {i}: {type(cfg_given_args[i])}, expected type {arg_conv_fn}")

                    def __process_part():
                        if not line_split[0] in UMDT_Parse.AVAILABLE_PART_CMDS:
                            raise UMDTParseException(
                                f"Part '{line_split[0]}' does not exist.")

                        part_name = line_split[0]
                        part_wanted_args = UMDT_Parse.AVAILABLE_PART_CMDS[line_split[0]]
                        part_given_args = line_split[1:] if len(
                            line_split) else []
                        if len(part_given_args) != len(part_wanted_args):
                            raise UMDTParseException("Unexpected number of given arguments to track command " +
                                                     f"'{part_name}': {len(part_given_args)}, expected {len(part_wanted_args)} ")

                        parsed_part = [part_name]
                        for i, arg_conv_fn in enumerate(part_wanted_args):
                            try:
                                parsed_part.append(arg_conv_fn(
                                    part_given_args[i].strip().replace('\n', '')))
                            except:
                                raise UMDTParseException("Unexpected argument type given to track command " +
                                                         f"'{part_name}', index {i}: {type(part_given_args[i])}, expected type {arg_conv_fn}")

                        state.data["parts"].append(parsed_part)

                    if line_split[0] == "cfg":
                        __process_cfg()
                    else:
                        __process_part()

        except FileNotFoundError:
            raise Exception("file not found")

        return state.data
