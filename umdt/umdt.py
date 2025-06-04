#!/bin/python3


from umdt.geometry import *
from umdt.visualize import *
from umdt.parse import *
import umdt.state as state
import logging
import sys
VERSION = "1.0.0"


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("umdt")


class UMDT(UMDT_Parse, UMDT_Visualize, UMDT_Geometry):
    pass


def __umdt_cmd_check(file_name):
    import json
    try:
        parsed = UMDT.parse(file_name)
        logger.info("the following is the parsed data: ")
        # print(json.dumps(parsed, indent=4))
        print(parsed)
        logger.info(f"{file_name}: parsed successfully!")
    except UMDTParseException as e:
        logger.error(f"{file_name}: failed parse check, reason:")
        logger.error(e)

    return True


def __umdt_cmd_visualize(file_name):
    import json
    try:
        parsed = UMDT.parse(file_name)
        logger.info(f"{file_name}: parsed successfully!")
        UMDT.visualize(parsed)
    except UMDTParseException as e:
        logger.error(f"{file_name}: failed parse check, reason:")
        logger.error(e)


def main():
    info = f"""UMDT v{VERSION}
umdt - A syntax checker, visualizer, and converter for UMDT files. Also can be used as a library.

Usage:
    umdt [command] [file]

Commands:
    check - Check the syntax of a UMDT file.
    visualize - Visualize a UMDT file in a graphical format.
    convert - Convert a UMDT file to another format.
    help - Display this message.

Examples:
    umdt check track.umdt
    umdt visualize track.umdt
    umdt convert track.umdt track.urdf
    """

    if len(sys.argv) < 3:
        print(info)
        sys.exit(-1)

    command = sys.argv[1]

    if command == "check":
        __umdt_cmd_check(sys.argv[2])
    elif command == "visualize":
        if not __umdt_cmd_check(sys.argv[2]):
            print("UMDT file is invalid.")
            sys.exit(-2)
        __umdt_cmd_visualize(sys.argv[2])
    elif command == "convert":
        pass
    elif command == "help":
        print(info)
