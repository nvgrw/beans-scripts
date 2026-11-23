#!/usr/bin/env python3

# Copyright (c) 2025 Niklas Vangerow
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import io
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import beancount as bn
from beancount.core import data as bn_data
from dateutil.relativedelta import relativedelta

_MAIN_CCY = "USD"
_TRADE_DATE_META = "trade-date"
_CENT = Decimal("0.01")
_WASH_WINDOW_DAYS = 30


class LotState(Enum):
    PENDING = 1  # Future buy not yet reached in chronological replay
    OPEN = 2     # Currently held in inventory
    CLOSED = 3   # Sold / disposed


@dataclass
class Lot:
    """An individual tax lot (or split sub-lot) with uniform basis & holding period."""

    lot_id: int
    purchase_id: int
    buy_seq: int
    orig_cost: bn.Cost
    acquired_date: date
    holding_date: date
    shares: Decimal
    orig_unit_cost: Decimal
    basis_adj_per_share: Decimal = Decimal(0)
    is_replacement: bool = False
    state: LotState = LotState.PENDING

    @property
    def adjusted_unit_cost(self) -> Decimal:
        return self.orig_unit_cost + self.basis_adj_per_share

    @property
    def total_orig_basis(self) -> Decimal:
        return self.shares * self.orig_unit_cost

    @property
    def total_basis_adjustment(self) -> Decimal:
        return self.shares * self.basis_adj_per_share

    @property
    def total_adjusted_basis(self) -> Decimal:
        return self.shares * self.adjusted_unit_cost


@dataclass
class _SellLeg:
    orig_cost: bn.Cost
    shares: Decimal
    sell_price: Decimal


@dataclass
class _Event:
    event_date: date
    is_sell: bool
    ledger_seq: int
    txn: bn.dtypes.Transaction
    buy_lots: List[Lot]
    sell_legs: List[_SellLeg]


@dataclass
class _SoldChunk:
    lot: Lot
    sell_price: Decimal


@dataclass
class Form8949Entry:
    property: bn.Amount
    acquired: date
    orig_acquired: date
    sold: date
    proceeds: bn.Amount
    orig_basis: bn.Amount
    basis_adjustment: bn.Amount
    basis: bn.Amount
    code_b: bool
    code_w: bool
    disallowed_loss: bn.Amount
    gl_adjustment: bn.Amount
    gl: bn.Amount
    is_long_term: bool


@dataclass
class WashSaleResult:
    entries: List[Form8949Entry]
    open_lots: List[Lot]


def _quantize_cents(val: Decimal) -> Decimal:
    return val.quantize(_CENT, rounding=ROUND_HALF_UP)


def _format_money(val: Decimal) -> str:
    q = _quantize_cents(val)
    if q == Decimal("-0.00"):
        q = Decimal("0.00")
    return f"{q:.2f}"


def is_long_term(holding_date: date, sold_date: date) -> bool:
    """IRC § 1222: long-term if held MORE than 1 year."""
    return sold_date > holding_date + relativedelta(years=1)


def trade_date(txn: bn.dtypes.Transaction) -> date:
    """Return the trade-date metadata override if present, otherwise txn.date."""
    date_override = txn.meta.get(_TRADE_DATE_META, None)
    if date_override is not None and isinstance(date_override, date):
        return date_override
    return txn.date


def get_commodity_transactions(
    entries: Sequence[bn.dtypes.Transaction], commodity: str
) -> List[bn.dtypes.Transaction]:
    return [
        entry
        for entry in entries
        if any(p.units.currency == commodity for p in entry.postings)
    ]


def _is_in_kind_transfer(postings: Sequence[bn.Posting]) -> bool:
    """True if postings net to zero per Cost and have no sale price (e.g., brokerage transfer)."""
    if not postings:
        return False
    if any(p.price is not None for p in postings):
        return False
    net_by_cost: Dict[Optional[bn.Cost], Decimal] = defaultdict(Decimal)
    for p in postings:
        net_by_cost[p.cost] += p.units.number
    return all(net == Decimal(0) for net in net_by_cost.values())


def _build_events(
    txns: Sequence[bn.dtypes.Transaction], commodity: str
) -> Tuple[List[_Event], List[Lot], int]:
    """Parse Beancount transactions into chronological Buy/Sell events and pre-created Lots."""
    events: List[_Event] = []
    all_lots: List[Lot] = []
    next_lot_id = 1
    next_purchase_id = 1

    for ledger_seq, txn in enumerate(txns):
        postings = [p for p in txn.postings if p.units.currency == commodity]
        if not postings:
            continue
        if _is_in_kind_transfer(postings):
            continue

        buys = [p for p in postings if p.units.number > bn.ZERO]
        sells = [p for p in postings if p.units.number < bn.ZERO]

        if buys and sells:
            raise ValueError(
                f"Found {len(sells)} sale postings and {len(buys)} buy postings in "
                f"transaction {txn}. Mixed buy/sell transactions (other than balanced "
                f"in-kind transfers) are not supported."
            )

        tdate = trade_date(txn)

        if buys:
            buy_lots: List[Lot] = []
            # Determine effective event date for chronological replay.
            # If the user entered a settlement date on the txn and an earlier cost.date,
            # use cost.date; if cost.date is later than txn.date (as in synthetic tests),
            # cap at tdate so the lot is open before subsequent ledger sales.
            buy_dates: List[date] = []
            for p in buys:
                if p.cost is None or not isinstance(p.cost, bn.Cost):
                    raise ValueError(f"Buy posting requires full Cost: {p}")
                if p.cost.currency != _MAIN_CCY:
                    raise ValueError(
                        f"Expected cost currency {_MAIN_CCY}, got {p.cost.currency} in {p}"
                    )
                acq_date = p.cost.date
                if acq_date is None:
                    raise ValueError(f"Buy posting cost requires a date: {p}")
                lot = Lot(
                    lot_id=next_lot_id,
                    purchase_id=next_purchase_id,
                    buy_seq=ledger_seq,
                    orig_cost=p.cost,
                    acquired_date=acq_date,
                    holding_date=acq_date,
                    shares=p.units.number,
                    orig_unit_cost=p.cost.number,
                    state=LotState.PENDING,
                )
                next_lot_id += 1
                next_purchase_id += 1
                buy_lots.append(lot)
                all_lots.append(lot)
                buy_dates.append(min(tdate, acq_date))

            events.append(
                _Event(
                    event_date=min(buy_dates),
                    is_sell=False,
                    ledger_seq=ledger_seq,
                    txn=txn,
                    buy_lots=buy_lots,
                    sell_legs=[],
                )
            )
        elif sells:
            sell_legs: List[_SellLeg] = []
            for p in sells:
                if p.cost is None or not isinstance(p.cost, bn.Cost):
                    raise ValueError(f"Sale posting requires full Cost: {p}")
                if p.cost.currency != _MAIN_CCY:
                    raise ValueError(
                        f"Expected cost currency {_MAIN_CCY}, got {p.cost.currency} in {p}"
                    )
                if p.price is None:
                    raise ValueError(f"Sale posting requires a price (@ ...): {p}")
                if p.price.currency != _MAIN_CCY:
                    raise ValueError(
                        f"Expected price currency {_MAIN_CCY}, got {p.price.currency} in {p}"
                    )
                sell_legs.append(
                    _SellLeg(
                        orig_cost=p.cost,
                        shares=-p.units.number,
                        sell_price=p.price.number,
                    )
                )
            events.append(
                _Event(
                    event_date=tdate,
                    is_sell=True,
                    ledger_seq=ledger_seq,
                    txn=txn,
                    buy_lots=[],
                    sell_legs=sell_legs,
                )
            )

    # Sort chronologically by effective trade date; on the same day process buys before sells
    # so same-day open-and-close trades work even when trade-date overrides reorder entries.
    events.sort(key=lambda e: (e.event_date, 1 if e.is_sell else 0, e.ledger_seq))
    return events, all_lots, next_lot_id


def _split_lot(lot: Lot, take_shares: Decimal, all_lots: List[Lot], next_lot_id: int) -> Tuple[Lot, int]:
    """Split `take_shares` off `lot`, returning the new taken Lot and updated next_lot_id.

    `lot` retains the remaining `lot.shares - take_shares` shares in place.
    """
    assert Decimal(0) < take_shares < lot.shares
    lot.shares -= take_shares
    taken = Lot(
        lot_id=next_lot_id,
        purchase_id=lot.purchase_id,
        buy_seq=lot.buy_seq,
        orig_cost=lot.orig_cost,
        acquired_date=lot.acquired_date,
        holding_date=lot.holding_date,
        shares=take_shares,
        orig_unit_cost=lot.orig_unit_cost,
        basis_adj_per_share=lot.basis_adj_per_share,
        is_replacement=lot.is_replacement,
        state=lot.state,
    )
    all_lots.append(taken)
    return taken, next_lot_id + 1


def calculate_wash_sales(
    entries: bn.Directives,
    commodity: str,
    unadjusted_basis_code_b: bool = False,
) -> WashSaleResult:
    """Run IRS IRC § 1091 & § 1223(3) wash-sale calculations across all entries for `commodity`."""
    txns = get_commodity_transactions(
        list(bn_data.sorted(bn.filter_txns(entries))), commodity
    )
    events, all_lots, next_lot_id = _build_events(txns, commodity)

    # Map purchase_id -> list of lots belonging to that buy event (updated on split)
    buy_event_purchases: Dict[int, List[Lot]] = defaultdict(list)
    for lot in all_lots:
        buy_event_purchases[lot.purchase_id].append(lot)

    form_entries: List[Form8949Entry] = []

    for event in events:
        if not event.is_sell:
            # Transition all sub-lots originating from this buy event from PENDING -> OPEN.
            for orig_lot in event.buy_lots:
                for sub_lot in buy_event_purchases[orig_lot.purchase_id]:
                    if sub_lot.state == LotState.PENDING:
                        sub_lot.state = LotState.OPEN
            continue

        # ---- PROCESS SALE EVENT ----
        # Step 1: Close ALL sold shares for this transaction BEFORE matching wash sales
        # so that a lot sold in this transaction cannot act as a replacement share for
        # another lot sold in the same transaction.
        sold_chunks: List[_SoldChunk] = []
        for leg in event.sell_legs:
            remaining = leg.shares
            matching_open = sorted(
                (
                    lot
                    for lot in all_lots
                    if lot.state == LotState.OPEN
                    and lot.shares > Decimal(0)
                    and lot.orig_cost == leg.orig_cost
                ),
                # FIFO by effective holding date (tacked washed sub-lots first), then acquisition/buy order.
                key=lambda l: (l.holding_date, l.acquired_date, l.buy_seq, l.lot_id),
            )
            for lot in matching_open:
                if remaining <= Decimal(0):
                    break
                if lot.shares <= remaining:
                    lot.state = LotState.CLOSED
                    sold_chunks.append(_SoldChunk(lot=lot, sell_price=leg.sell_price))
                    remaining -= lot.shares
                else:
                    taken, next_lot_id = _split_lot(lot, remaining, all_lots, next_lot_id)
                    buy_event_purchases[taken.purchase_id].append(taken)
                    taken.state = LotState.CLOSED
                    sold_chunks.append(_SoldChunk(lot=taken, sell_price=leg.sell_price))
                    remaining = Decimal(0)

            if remaining > Decimal(0):
                raise ValueError(
                    f"Insufficient open shares of {commodity} with cost {leg.orig_cost} "
                    f"on {event.event_date} (short by {remaining} shares) in {event.txn}"
                )

        # Step 2: Process each sold chunk in chronological order of acquisition
        # (Treas. Reg. § 1.1091-1(c)).
        sold_chunks.sort(
            key=lambda c: (c.lot.acquired_date, c.lot.buy_seq, c.lot.lot_id)
        )

        sale_dt = event.event_date
        oldest_window = sale_dt - timedelta(days=_WASH_WINDOW_DAYS)
        newest_window = sale_dt + timedelta(days=_WASH_WINDOW_DAYS)

        for chunk in sold_chunks:
            lot = chunk.lot
            proceeds = lot.shares * chunk.sell_price
            orig_basis = lot.total_orig_basis
            basis_adj = lot.total_basis_adjustment
            adjusted_basis = lot.total_adjusted_basis
            raw_gl = proceeds - adjusted_basis

            disallowed_loss = Decimal(0)
            code_w = False

            if raw_gl < Decimal(0):
                loss_per_share = (-raw_gl) / lot.shares
                held_days = max(0, (sale_dt - lot.holding_date).days)

                # Eligible replacement shares: open now OR pending future acquisition
                # within [sale_dt - 30, sale_dt + 30], not already used as replacement,
                # and not from the same purchase as the sold shares.
                # Ordered strictly by chronological acquisition date (Treas. Reg. § 1.1091-1(c)-(d)).
                candidates = sorted(
                    (
                        cand
                        for cand in all_lots
                        if cand.state in (LotState.OPEN, LotState.PENDING)
                        and cand.shares > Decimal(0)
                        and not cand.is_replacement
                        and cand.purchase_id != lot.purchase_id
                        and oldest_window <= cand.acquired_date <= newest_window
                    ),
                    key=lambda c: (c.acquired_date, c.buy_seq, c.lot_id),
                )

                remaining_to_wash = lot.shares
                for cand in candidates:
                    if remaining_to_wash <= Decimal(0):
                        break
                    if cand.shares <= remaining_to_wash:
                        matched = cand
                    else:
                        matched, next_lot_id = _split_lot(
                            cand, remaining_to_wash, all_lots, next_lot_id
                        )
                        buy_event_purchases[matched.purchase_id].append(matched)

                    matched.is_replacement = True
                    matched.basis_adj_per_share += loss_per_share
                    matched.holding_date = matched.holding_date - timedelta(days=held_days)

                    disallowed_loss += matched.shares * loss_per_share
                    remaining_to_wash -= matched.shares
                    code_w = True

            # Form 8949 reporting:
            # Always satisfies Col(h) == Col(d) - Col(e) + Col(g).
            if unadjusted_basis_code_b:
                code_b = basis_adj != Decimal(0)
                col_e_basis = orig_basis
                col_g_adj = disallowed_loss - basis_adj
            else:
                code_b = False
                col_e_basis = adjusted_basis
                col_g_adj = disallowed_loss

            final_gl = proceeds - col_e_basis + col_g_adj

            form_entries.append(
                Form8949Entry(
                    property=bn.amount.Amount(lot.shares, commodity),
                    acquired=lot.holding_date,
                    orig_acquired=lot.acquired_date,
                    sold=sale_dt,
                    proceeds=bn.amount.Amount(proceeds, _MAIN_CCY),
                    orig_basis=bn.amount.Amount(orig_basis, _MAIN_CCY),
                    basis_adjustment=bn.amount.Amount(basis_adj, _MAIN_CCY),
                    basis=bn.amount.Amount(col_e_basis, _MAIN_CCY),
                    code_b=code_b,
                    code_w=code_w,
                    disallowed_loss=bn.amount.Amount(disallowed_loss, _MAIN_CCY),
                    gl_adjustment=bn.amount.Amount(col_g_adj, _MAIN_CCY),
                    gl=bn.amount.Amount(final_gl, _MAIN_CCY),
                    is_long_term=is_long_term(lot.holding_date, sale_dt),
                )
            )

    open_lots = sorted(
        (lot for lot in all_lots if lot.state == LotState.OPEN and lot.shares > Decimal(0)),
        key=lambda l: (l.acquired_date, l.buy_seq, l.lot_id),
    )
    return WashSaleResult(entries=form_entries, open_lots=open_lots)


def calculate_8949_entries(
    entries: bn.Directives,
    commodity: str,
    year: Optional[int] = None,
    unadjusted_basis_code_b: bool = False,
) -> List[Form8949Entry]:
    """Calculate Form 8949 entries for `commodity`, optionally filtered to sales in `year`."""
    result = calculate_wash_sales(
        entries, commodity, unadjusted_basis_code_b=unadjusted_basis_code_b
    )
    if year is None:
        return result.entries
    return [e for e in result.entries if e.sold.year == year]


_COLUMNS = (
    "(a) Description of property",
    "(b) Date acquired",
    "(c) Date sold",
    "(d) Proceeds",
    "(e) Cost or other basis",
    "(f) Code(s)",
    "(g) Adjustment",
    "(h) Gain or (loss)",
)


def _entry_to_row(entry: Form8949Entry) -> List[str]:
    codes = ("B" if entry.code_b else "") + ("W" if entry.code_w else "")
    return [
        f"{entry.property.number} SHARES OF {entry.property.currency}",
        entry.acquired.strftime("%m/%d/%y"),
        entry.sold.strftime("%m/%d/%y"),
        _format_money(entry.proceeds.number),
        _format_money(entry.basis.number),
        codes,
        _format_money(entry.gl_adjustment.number),
        _format_money(entry.gl.number),
    ]


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]], right_align: Iterable[int] = ()) -> str:
    right = set(right_align)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i in right:
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return "  ".join(parts)

    lines = [
        fmt_row(headers),
        "  ".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def format_report_table(entries: Sequence[Form8949Entry], open_lots: Sequence[Lot] = ()) -> str:
    """Format Form 8949 entries split into Part I (Short-Term) and Part II (Long-Term)."""
    if not entries and not open_lots:
        return "No sales or open lots found."

    sections: List[str] = []
    short_term = [e for e in entries if not e.is_long_term]
    long_term = [e for e in entries if e.is_long_term]

    for title, group in (
        ("Form 8949 Part I — Short-Term", short_term),
        ("Form 8949 Part II — Long-Term", long_term),
    ):
        if not group:
            continue
        rows = [_entry_to_row(e) for e in group]
        tot_proceeds = sum((e.proceeds.number for e in group), Decimal(0))
        tot_basis = sum((e.basis.number for e in group), Decimal(0))
        tot_adj = sum((e.gl_adjustment.number for e in group), Decimal(0))
        tot_gl = sum((e.gl.number for e in group), Decimal(0))
        rows.append(
            [
                "TOTAL",
                "",
                "",
                _format_money(tot_proceeds),
                _format_money(tot_basis),
                "",
                _format_money(tot_adj),
                _format_money(tot_gl),
            ]
        )
        sections.append(f"{title}\n{_render_table(_COLUMNS, rows, right_align=(3, 4, 6, 7))}")

    if entries:
        grand_gl = sum((e.gl.number for e in entries), Decimal(0))
        grand_disallowed = sum((e.disallowed_loss.number for e in entries), Decimal(0))
        sections.append(
            f"Grand Total Reported Gain/(Loss): {_format_money(grand_gl)} {_MAIN_CCY} "
            f"(Disallowed Wash Losses: {_format_money(grand_disallowed)} {_MAIN_CCY})"
        )

    if open_lots:
        lot_headers = (
            "Shares",
            "Orig Acquired",
            "Holding Date",
            "Orig Unit Cost",
            "Wash Adj/sh",
            "Adj Basis",
            "Replacement?",
        )
        lot_rows = [
            [
                str(lot.shares),
                lot.acquired_date.strftime("%m/%d/%y"),
                lot.holding_date.strftime("%m/%d/%y"),
                _format_money(lot.orig_unit_cost),
                _format_money(lot.basis_adj_per_share),
                _format_money(lot.total_adjusted_basis),
                "Yes" if lot.is_replacement else "No",
            ]
            for lot in open_lots
        ]
        sections.append(
            f"Remaining Open Lots\n{_render_table(lot_headers, lot_rows, right_align=(0, 3, 4, 5))}"
        )

    return "\n\n".join(sections)


def format_report_csv(entries: Sequence[Form8949Entry]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Term", *_COLUMNS])
    for e in entries:
        writer.writerow(["Long-Term" if e.is_long_term else "Short-Term", *_entry_to_row(e)])
    return buf.getvalue()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate US Form 8949 wash sales from a Beancount ledger."
    )
    parser.add_argument("filename", help="Path to the Beancount ledger file")
    parser.add_argument("commodity", help="Commodity/ticker symbol to analyze (e.g., STONK)")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter Form 8949 output to sales in this tax year (full history is still scanned for wash windows)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--unadjusted-basis",
        action="store_true",
        help="Report original unadjusted basis in Col (e) with Code B and negative adjustment in Col (g)",
    )
    args = parser.parse_args(argv)

    entries, errors, _ = bn.load_file(args.filename)
    if errors:
        print(errors, file=sys.stderr)
        return 1

    result = calculate_wash_sales(
        entries, args.commodity, unadjusted_basis_code_b=args.unadjusted_basis
    )
    filtered = (
        [e for e in result.entries if e.sold.year == args.year]
        if args.year is not None
        else result.entries
    )

    if args.format == "csv":
        print(format_report_csv(filtered), end="")
    else:
        print(format_report_table(filtered, open_lots=result.open_lots))
    return 0


if __name__ == "__main__":
    sys.exit(main())
