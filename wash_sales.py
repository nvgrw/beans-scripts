#!/usr/bin/env python3

from collections import OrderedDict, defaultdict
import copy
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
import sys
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
from enum import Enum
import pandas as pd
import beancount as bn
from beancount.core import inventory as bn_inventory
from beancount.core import position as bn_position
from beancount.core import data as bn_data


def get_commodity_transactions(
    entries: List[bn.dtypes.Transaction], commodity: str
) -> List[bn.dtypes.Transaction]:
    return [
        entry
        for entry in entries
        if any(p.units.currency == commodity for p in entry.postings)
    ]


_MAIN_CCY = "USD"


@dataclass
class Form8949Entry:
    property: bn.Amount
    acquired: Set[date]
    sold: date
    proceeds: bn.Amount
    basis: bn.Amount
    code_b: bool
    code_w: bool
    gl_adjustment: bn.Amount
    gl: bn.Amount


def main(filename: str, commodity: str):
    entries, errors, option_map = bn.load_file(filename)
    if len(errors) != 0:
        print(errors, file=sys.stderr)
        sys.exit(1)

    txns: List[bn.dtypes.Transaction] = get_commodity_transactions(
        list(bn_data.sorted(bn.filter_txns(entries))), commodity
    )

    inv = bn.Inventory()
    cost_to_basis_adjustment: Dict[bn.Cost, Decimal] = dict()
    for txn in txns:
        # Count the number of postings that are sales and non-sales.
        sale_count = 0
        non_sale_count = 0

        inv_prev = bn.Inventory()
        cost_to_sell_price: Dict[bn.Cost, Decimal] = dict()
        for posting in txn.postings:
            if posting.units.currency != commodity:
                continue
            prev, match = inv.add_position(posting)
            assert match != bn_inventory.MatchResult.IGNORED, "IGNORED not supported."
            assert match != bn_inventory.MatchResult.AUGMENTED, (
                "AUGMENTED not supported."
            )

            if match == bn_inventory.MatchResult.CREATED:
                # Not a sale.
                non_sale_count += 1
                continue

            assert match == bn_inventory.MatchResult.REDUCED, "Expected REDUCED."
            assert posting.price is not None, "All postings require prices."
            inv_prev.add_position(prev)
            cost_to_sell_price[prev.cost] = posting.price.number
            sale_count += 1

        if sale_count > 0 and non_sale_count > 0:
            raise ValueError(
                "Found {} sale counts and {} non-sale counts in transaction {}. While transactions of this kind are valid in beancount, the wash sale analysis script does not support mixed buys/sales because it makes the logic more complicated.".format(
                    sale_count, non_sale_count, str(txn)
                )
            )
        if sale_count == 0:
            continue  # Not a sale.

        # ---- PROCESS THE SALE ----
        num_shares = Decimal()
        basis = Decimal()
        b_adjustment = Decimal()
        proceeds = Decimal()
        acquisition_dates: Set[date] = set()
        for position in inv_prev.get_positions():
            assert position.cost.currency == _MAIN_CCY
            num_shares += position.units.number
            basis += position.units.number * position.cost.number
            b_adjustment += cost_to_basis_adjustment.get(position.cost, Decimal())

            price: Optional[Decimal] = cost_to_sell_price.get(position.cost, None)
            assert price is not None, "No price for sale of position {}".format(
                position
            )
            proceeds += position.units.number * price
            acquisition_dates.add(position.cost.date)

        # if b_adjustment is set, set B flag.
        # apply b_adjustment.
        basis += b_adjustment

        e = Form8949Entry(
            property=bn.amount.Amount(num_shares, commodity),
            acquired=acquisition_dates,
            sold=txn.date,
            proceeds=bn.amount.Amount(proceeds, _MAIN_CCY),
            basis=bn.amount.Amount(basis, _MAIN_CCY),
            code_b=b_adjustment > bn.ZERO,
            code_w=False,
            gl_adjustment=bn.amount.Amount(bn.ZERO, _MAIN_CCY),
            gl=bn.amount.Amount(proceeds - basis, _MAIN_CCY),
        )
        print(e)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: {} filename.beancount COMMODITY".format(sys.argv[0]),
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
