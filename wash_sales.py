#!/usr/bin/env python3

import sys
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import beancount as bn
from beancount.core import inventory as bn_inventory
from beancount.core import data as bn_data

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


def get_commodity_transactions(
    entries: List[bn.dtypes.Transaction], commodity: str
) -> List[bn.dtypes.Transaction]:
    return [
        entry
        for entry in entries
        if any(p.units.currency == commodity for p in entry.postings)
    ]


def find_replacement_shares(
    txns: List[bn.dtypes.Transaction],
    sale_txn: bn.dtypes.Transaction,
    inv: bn.Inventory,
    inv_wash: bn.Inventory,
    inv_sold: bn.Inventory,
    commodity: str,
) -> List[Tuple[bn.Position, Decimal]]:
    oldest_date = sale_txn.date - timedelta(days=30)
    newest_date = sale_txn.date + timedelta(days=30)

    # In this function Tuple[Pos, Pos] = replacement, original number of shares.
    # This way cb adjustments make sense.

    # Find all positions within the window.
    pos_within_window: List[Tuple[bn.Position, Decimal]] = list()
    sale_index = txns.index(sale_txn)
    for i in range(sale_index + 1, len(txns)):
        txn = txns[i]
        for posting in txn.postings:
            if posting.units.currency != commodity:
                continue
            if posting.units.number > bn.ZERO and posting.cost.date <= newest_date:
                # A buy.
                pos_within_window.append(
                    (
                        bn.Position(units=posting.units, cost=posting.cost),
                        posting.units.number,
                    )
                )
    for position in inv.get_positions():
        if position.cost.date >= oldest_date and position.cost.date <= newest_date:
            pos_within_window.append((position, position.units.number))

    # Exclude those that acted as replacement shares.
    pos_not_replaced: List[Tuple[bn.Position, bn.Position]] = list()
    for position, orig_num_shares in pos_within_window:
        key = (position.units.currency, position.cost)
        if key not in inv_wash:
            pos_not_replaced.append((position, orig_num_shares))
            continue
        wash_position: bn.Position = inv_wash[key]
        if position.units.number > wash_position.units.number:
            pos_not_replaced.append(
                (
                    bn.Position(
                        bn.amount.sub(position.units, wash_position.units),
                        position.cost,
                    ),
                    orig_num_shares,
                )
            )

    pos_candidates: List[Tuple[bn.Position, Decimal]] = list()
    for outstanding_pos in inv_sold.get_positions():
        # Remainder for next matching position.
        remaining_pos_not_replaced: List[Tuple[bn.Position, Decimal]] = list()

        outstanding_shares = Decimal(outstanding_pos.units.number)
        for position, orig_num_shares in sorted(
            pos_not_replaced, key=lambda t: t[0].cost
        ):
            position: bn.Position
            if outstanding_shares <= bn.ZERO or position.cost == outstanding_pos.cost:
                # If nothing left or same cost, don't have a candidate.
                remaining_pos_not_replaced.append((position, orig_num_shares))
                continue
            if position.units.number <= outstanding_shares:
                pos_candidates.append((position, orig_num_shares))
            else:
                pos_candidates.append(
                    (
                        bn.Position(
                            units=bn.amount.Amount(
                                outstanding_shares, position.units.currency
                            ),
                            cost=position.cost,
                        ),
                        orig_num_shares,
                    )
                )
                remaining_pos_not_replaced.append(
                    (
                        bn.Position(
                            units=bn.amount.Amount(
                                position.units.number - outstanding_shares,
                                position.units.currency,
                            ),
                            cost=position.cost,
                        ),
                        orig_num_shares,
                    )
                )
            outstanding_shares -= position.units.number

        pos_not_replaced = remaining_pos_not_replaced
    return pos_candidates


# Applies the basis adjustment to the replacement shares, marks them, and returns the GL adjustment for this sale.
def apply_replacement_shares(
    inv_wash: bn.Inventory,
    cost_to_basis_adjustment_sh: Dict[bn.Cost, Decimal],
    ordered_available_replacement_shares: List[Tuple[bn.Position, Decimal]],
    num_shares: Decimal,
    gl: Decimal,
) -> Decimal:
    # Disallowed loss/sh
    gl_sh = gl / num_shares
    gl_adjustment = Decimal()
    for position, orig_num_shares in ordered_available_replacement_shares:
        # Update record of replaced shares.
        inv_wash.add_position(position)
        # Amount to adjust for this particular replacement position.
        cost_to_basis_adjustment_sh[position.cost] -= gl_sh * (
            position.units.number / orig_num_shares
        )
        # Adjust the g/l of the wash sale.
        gl_adjustment -= position.units.number * gl_sh
    return gl_adjustment


def entries_to_dataframe(entries: List[Form8949Entry]) -> Optional[pd.DataFrame]:
    if len(entries) == 0:
        return None

    def entry_to_row(entry: Form8949Entry) -> List[str]:
        row = [""] * 8
        row[0] = "{} SHARES OF {}".format(
            entry.property.number, entry.property.currency
        )
        if len(entry.acquired) > 1:
            row[1] = "VARIOUS"
        else:
            row[1] = list(entry.acquired)[0].strftime("%m/%d/%y")
        row[2] = entry.sold.strftime("%m/%d/%y")
        row[3] = str(entry.proceeds.number.quantize(Decimal("0.00")))
        row[4] = str(entry.basis.number.quantize(Decimal("0.00")))
        row[5] = ("B" if entry.code_b else "") + ("W" if entry.code_w else "")
        row[6] = str(entry.gl_adjustment.number.quantize(Decimal("0.00")))
        row[7] = str(
            bn.amount.add(entry.gl, entry.gl_adjustment).number.quantize(
                Decimal("0.00")
            )
        )
        return row

    return pd.DataFrame(
        data=(entry_to_row(entry) for entry in entries),
        columns=(
            "(a) Description of property",
            "(b) Date acquired",
            "(c) Date sold or disposed of",
            "(d) Proceeds",
            "(e) Cost or other basis",
            "(f) Code(s)",
            "(g) Amount of adjustment",
            "(h) Gain or (loss)",
        ),
    )


def calculate_8949_entries(
    entries: bn.Directives, commodity: str
) -> List[Form8949Entry]:
    txns: List[bn.dtypes.Transaction] = get_commodity_transactions(
        list(bn_data.sorted(bn.filter_txns(entries))), commodity
    )

    # The current inventory.
    inv = bn.Inventory()
    # A record of replacement share costs and how many.
    inv_wash = bn.Inventory()
    cost_to_basis_adjustment_sh: Dict[bn.Cost, Decimal] = defaultdict(Decimal)
    form_entries: List[Form8949Entry] = list()
    for txn in txns:
        # Count the number of postings that are sales and non-sales.
        sale_count = 0
        non_sale_count = 0

        inv_sold = bn.Inventory()
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
            inv_sold.add_position(bn.Position(units=-posting.units, cost=posting.cost))
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
        for position in inv_sold.get_positions():
            assert position.cost.currency == _MAIN_CCY
            num_shares += position.units.number
            basis += position.units.number * position.cost.number
            b_adjustment += (
                cost_to_basis_adjustment_sh.get(position.cost, Decimal())
                * position.units.number
            )

            price: Optional[Decimal] = cost_to_sell_price.get(position.cost, None)
            assert price is not None, "No price for sale of position {}".format(
                position
            )
            proceeds += position.units.number * price
            acquisition_dates.add(position.cost.date)

        # if b_adjustment is set, set B flag.
        # apply b_adjustment.
        basis += b_adjustment
        gl_adjustment = Decimal()
        gl: Decimal = proceeds - basis

        code_w = False
        if gl < bn.ZERO:  # Possible wash sale
            ordered_available_replacement_shares = find_replacement_shares(
                txns, txn, inv, inv_wash, inv_sold, commodity
            )
            if len(ordered_available_replacement_shares) > 0:
                # A wash sale with a full or partial replacement.
                code_w = True
                gl_adjustment = apply_replacement_shares(
                    inv_wash,
                    cost_to_basis_adjustment_sh,
                    ordered_available_replacement_shares,
                    num_shares,
                    gl,
                )

        form_entries.append(
            Form8949Entry(
                property=bn.amount.Amount(num_shares, commodity),
                acquired=acquisition_dates,
                sold=txn.date,
                proceeds=bn.amount.Amount(proceeds, _MAIN_CCY),
                basis=bn.amount.Amount(basis, _MAIN_CCY),
                code_b=b_adjustment > bn.ZERO,
                code_w=code_w,
                gl_adjustment=bn.amount.Amount(gl_adjustment, _MAIN_CCY),
                gl=bn.amount.Amount(gl, _MAIN_CCY),
            )
        )

    return form_entries


def main(filename: str, commodity: str):
    entries, errors, option_map = bn.load_file(filename)
    if len(errors) != 0:
        print(errors, file=sys.stderr)
        sys.exit(1)

    form_entries: List[Form8949Entry] = calculate_8949_entries(entries, commodity)
    print(entries_to_dataframe(form_entries))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: {} filename.beancount COMMODITY".format(sys.argv[0]),
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
