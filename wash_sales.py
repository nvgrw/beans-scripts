#!/usr/bin/env python3

from collections import OrderedDict, defaultdict
import copy
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
import sys
from typing import Any, Dict, Generator, List, Optional, Tuple
import beancount as bn
import beancount.core.data as bn_data
from beancount.core import inventory as bn_inventory
from beancount.core import position as bn_position
from enum import Enum
import pandas as pd

def main(filename: str, commodity: str):
    entries, errors, option_map = bn.load_file(filename)
    if len(errors) != 0:
        print(errors, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: {} filename.beancount COMMODITY".format(sys.argv[0]),
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
