#!/usr/bin/env python3

from collections import OrderedDict
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

# Simplifying assumptions:
#
# All lots have dates indicating the purchase date.
# Held at cost.
# Sales are when the inventory is reduced. Sale date is taken as the txn date.
# Wire in some kind of reported sale date / ETP-override?

_MAIN_CCY: str = "USD"


class TxnType(Enum):
    SELL = 1
    BUY = 2


# A useful representation for intermediate WSA state.
@dataclass
class TxnEntry:
    date: date
    type: TxnType
    txn: bn.dtypes.Transaction
    filtered_postings: list[bn.Posting] = field(default_factory=list)
    pre_inv: bn.Inventory = field(default_factory=bn.Inventory)
    post_inv: bn.Inventory = field(default_factory=bn.Inventory)
    basis_adjustment: bn.Amount = field(
        default_factory=lambda: bn.Amount(bn.ZERO, _MAIN_CCY)
    )
    # v BUY ONLY v
    shares: Decimal = field(default_factory=lambda: Decimal(bn.ZERO))
    replaced: Decimal = field(default_factory=lambda: Decimal(bn.ZERO))
    # v SELL ONLY v
    code_w: bool = False
    code_b: bool = False


def preprocess(
    filtered_txns: List[bn.dtypes.Transaction], commodity: str
) -> List[TxnEntry]:
    inv = bn.Inventory()
    txn_states: list[TxnEntry] = list()
    for txn in filtered_txns:
        # We need to classify this transaction as a "buy" or a "sell"
        # transaction. We don't allow both in the same transaction.
        txn_type: Optional[TxnType] = None
        filtered_postings = [
            posting for posting in txn.postings if posting.units.currency == commodity
        ]
        shares: Decimal = bn.ZERO
        for posting in filtered_postings:
            assert posting.cost is not None, (
                "All commodity postings must be held at cost."
            )
            assert isinstance(posting.cost, bn.Cost), (
                "All postings require full cost information"
            )
            _, match_result = inv.add_position(posting)
            if match_result == bn_inventory.MatchResult.CREATED:
                if txn_type is None or txn_type == TxnType.BUY:
                    txn_type = TxnType.BUY
                else:
                    raise ValueError(
                        "txn_type was not BUY but found a creation or augmentation of the inventory."
                    )
            elif match_result == bn_inventory.MatchResult.REDUCED:
                if txn_type is None or txn_type == TxnType.SELL:
                    txn_type = TxnType.SELL
                else:
                    raise ValueError(
                        "txn_type was not SELL, but found a reduction of the inventory."
                    )
            else:
                raise ValueError("Don't know how to handle match result ", match_result, posting)
            shares += abs(posting.units.number)

        if txn_type is None or len(filtered_postings) == 0:
            # Neither, so skip.
            continue
        if txn_type == TxnType.BUY and len(filtered_postings) > 1:
            raise ValueError(
                "Buy transactions can only have one CCY posting (multiple lots cannot be opened in the same transaction)."
            )
        txn_states.append(
            TxnEntry(
                date=txn.date
                if txn_type == TxnType.SELL
                else filtered_postings[0].cost.date,
                type=txn_type,
                txn=txn,
                filtered_postings=filtered_postings,
                shares=shares,
            )
        )

    return txn_states


def basis_proceeds(
    cost_to_buy_txn: Dict[bn.Cost, TxnEntry], te: TxnEntry, adjust_basis: bool = False
) -> Tuple[bn.Amount, bn.Amount, bool]:
    assert te.type == TxnType.SELL, "basis_proceeds only works with SELL transactions."
    assert len(te.filtered_postings) > 0, (
        "sale needs at least one posting, else it's a no-op"
    )

    basis = bn.amount.Amount(bn.ZERO, _MAIN_CCY)
    proceeds = bn.amount.Amount(bn.ZERO, _MAIN_CCY)
    holding_adjusted_basis = False
    for posting in te.filtered_postings:
        basis = bn.amount.sub(basis, bn.get_cost(posting))
        # Apply adjustment
        buy_txn: TxnEntry = cost_to_buy_txn[posting.cost]
        assert buy_txn is not None, "Could not find buy transaction for posting"
        if buy_txn.basis_adjustment.number != bn.ZERO:
            basis = bn.amount.add(basis, buy_txn.basis_adjustment)
            holding_adjusted_basis = True
        proceeds = bn.amount.sub(
            proceeds, bn.amount.mul(posting.price, posting.units.number)
        )
    if adjust_basis:
        basis = bn.amount.add(basis, te.basis_adjustment)

    return basis, proceeds, holding_adjusted_basis


# Returns a list of BUY TxnEntry transactions that are candidates, and the
# number of shares remaining for the corresponding lots at the point of the
# sale.
def find_candidates(
    cost_to_buy_txn: Dict[bn.Cost, TxnEntry], subject: TxnEntry, txns: List[TxnEntry]
) -> Generator[TxnEntry, None, None]:
    start_date: date = subject.date - timedelta(days=30)
    end_date: date = subject.date + timedelta(days=30)
    subject_costs = {posting.cost for posting in subject.filtered_postings}

    # Records all lots and # of shares available to wash with the subject.
    inv = bn.Inventory()
    seen_subject = False
    for txn in txns:
        if txn.txn == subject.txn:
            seen_subject = True
            continue
        if not seen_subject or (seen_subject and txn.type == TxnType.BUY):
            # If subject has not been seen, record buy + sell txns in the inventory to know what is available.
            # If subject has been seen, only record buy txns to understand what can be washed with.
            # Augmentation is NOT SUPPORTED.
            for posting in txn.filtered_postings:
                prev_position, match_result = inv.add_position(posting)
                assert (
                    match_result != bn_inventory.MatchResult.AUGMENTED
                    and match_result != bn_inventory.MatchResult.IGNORED
                ), "Cost can neither be augmented nor ignored"

                # A deleted position is not reflected in the inventory, but we
                # need to record the number of shares.
                if (
                    match_result == bn_inventory.MatchResult.REDUCED
                    and prev_position is not None
                ):
                    curr_position = inv.get(
                        (posting.units.currency, prev_position.cost)
                    )
                    if curr_position is None:
                        cost_to_buy_txn[prev_position.cost].shares = Decimal()

    # Update buy txns with share counts that are still held or will be in the future.
    # TODO: This value should probably just be returned instead of overloading BUY.shares
    for position in inv.get_positions():
        cost_to_buy_txn[position.cost].shares = position.units.number

    for txn in txns:
        if txn.type != TxnType.BUY:
            continue  # Only wash with BUY txns and not subject (which is a SELL)
        if txn.date < start_date or txn.date > end_date:
            continue  # Transaction outside window
        if txn.shares <= bn.ZERO or txn.replaced >= txn.shares:
            continue  # No shares left
        assert len(txn.filtered_postings) == 1, "Each BUY should have one posting."

        posting = txn.filtered_postings[0]
        if posting.cost in subject_costs:
            continue  # Same lot doesn't wash

        yield txn


def calculate_washes(cost_to_buy_txn: Dict[bn.Cost, TxnEntry], txns: List[TxnEntry]):
    for sale in txns:
        if sale.type != TxnType.SELL:
            continue

        basis, proceeds, holding_adjusted_basis = basis_proceeds(
            cost_to_buy_txn, sale, adjust_basis=True
        )
        # B indicates that basis is not what was reported
        sale.code_b = holding_adjusted_basis

        gain: bn.Amount = bn.amount.sub(proceeds, basis)
        if gain.number >= bn.ZERO:  # can't wash a gain
            continue

        per_share_disallowed_gain = bn.Amount = bn.amount.div(gain, sale.shares)
        outstanding = sale.shares
        for candidate in find_candidates(cost_to_buy_txn, sale, txns):
            if outstanding == bn.ZERO:
                break  # Exhausted, all shares replaced.
            sale.code_w = True  # W indicates that this is a wash sale

            replaced = min(candidate.shares - candidate.replaced, outstanding)
            outstanding -= replaced
            candidate.replaced += replaced
            current_adjustment = bn.amount.Amount(
                per_share_disallowed_gain.number * replaced,
                per_share_disallowed_gain.currency,
            )
            sale.basis_adjustment = bn.amount.add(
                sale.basis_adjustment, current_adjustment
            )
            candidate.basis_adjustment = bn.amount.sub(
                candidate.basis_adjustment, current_adjustment
            )


def calculate_cost_to_buy_txn_entry(txns: List[TxnEntry]) -> Dict[bn.Cost, TxnEntry]:
    cost_to_buy_txn_entry: Dict[bn.Cost, TxnEntry] = dict()
    for txn in txns:
        if txn.type != TxnType.BUY:
            continue
        assert len(txn.filtered_postings) == 1, (
            "Buy transactions can only have one filtered posting"
        )
        posting = txn.filtered_postings[0]
        assert isinstance(posting.cost, bn.Cost), "Posting cost must be bn.Cost"
        cost_to_buy_txn_entry[posting.cost] = txn
    return cost_to_buy_txn_entry


def generate_8949(
    cost_to_buy_txn: Dict[bn.Cost, TxnEntry], commodity: str, txns: List[TxnEntry]
) -> pd.DataFrame:
    data: List[List[str]] = list()

    for txn in txns:
        if txn.type != TxnType.SELL:
            continue
        basis, proceeds, _ = basis_proceeds(cost_to_buy_txn, txn)
        gain: bn.Amount = bn.amount.sub(proceeds, basis)
        row = [""] * 8
        row[0] = "{} SHARES OF {}".format(txn.shares, commodity)
        if len(txn.filtered_postings) > 1:
            row[1] = "VARIOUS"
        else:
            row[1] = txn.filtered_postings[0].cost.date.strftime("%m/%d/%y")
        row[2] = txn.date.strftime("%m/%d/%y")
        row[3] = str(proceeds.number.quantize(Decimal("0.00")))
        row[4] = str(basis.number.quantize(Decimal("0.00")))
        row[5] = ("B" if txn.code_b else "") + ("W" if txn.code_w else "")
        row[6] = str((-txn.basis_adjustment.number).quantize(Decimal("0.00")))
        row[7] = str(
            bn.amount.sub(gain, txn.basis_adjustment).number.quantize(Decimal("0.00"))
        )
        data.append(row)

    return pd.DataFrame(
        data=data,
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


def main(filename: str, commodity: str):
    entries, errors, option_map = bn.load_file(filename)
    if len(errors) != 0:
        print(errors, file=sys.stderr)
        sys.exit(1)

    txns: List[bn.dtypes.Transaction] = bn_data.sorted(
        list(
            txn
            for txn in bn.filter_txns(entries)
            if any(posting.units.currency == commodity for posting in txn.postings)
        )
    )

    tes: List[TxnEntry] = preprocess(txns, commodity)
    cost_to_buy_txn_entry: Dict[bn.Cost, TxnEntry] = calculate_cost_to_buy_txn_entry(
        tes
    )
    calculate_washes(cost_to_buy_txn_entry, tes)
    # print("\n\n".join(str(te) for te in tes))
    print(generate_8949(cost_to_buy_txn_entry, commodity, tes))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: {} filename.beancount COMMODITY".format(sys.argv[0]),
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
