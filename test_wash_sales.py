from datetime import date
from decimal import Decimal
from pathlib import Path
import textwrap

import beancount as bn
from beancount import loader
import pytest

import wash_sales as ws

EXAMPLES_DIR = Path(__file__).parent / "examples"


def _load_str(ledger_text: str):
    full = textwrap.dedent(
        """\
        1970-01-01 commodity STONK
        1970-01-01 open Assets:Checking USD
        1970-01-01 open Assets:Brokerage:STONK STONK
        1970-01-01 open Assets:Brokerage2:STONK STONK
        1970-01-01 open Income:PnL USD
        """
    ) + textwrap.dedent(ledger_text)
    entries, errors, _ = loader.load_string(full)
    assert not errors, f"Beancount load errors: {errors}"
    return entries


def _load_example(name: str):
    entries, errors, _ = bn.load_file(str(EXAMPLES_DIR / name))
    assert not errors, f"Beancount load errors in {name}: {errors}"
    return entries


def _assert_conservation_invariant(result: ws.WashSaleResult):
    """Verify exact conservation of basis & Form 8949 arithmetic on every entry.

    1. Form 8949 row identity: Col(h) == Col(d) - Col(e) + Col(g)
    2. Global conservation:
         sum(reported G/L) - sum(open_lots.total_basis_adjustment)
         == sum(proceeds - orig_basis)
       (Every dollar of disallowed loss either increases an open lot's basis or was
       already realized when that replacement lot was sold.)
    """
    for e in result.entries:
        assert e.gl.number == e.proceeds.number - e.basis.number + e.gl_adjustment.number

    total_reported_gl = sum((e.gl.number for e in result.entries), Decimal(0))
    total_open_basis_adj = sum((l.total_basis_adjustment for l in result.open_lots), Decimal(0))
    total_economic_sold_gl = sum(
        (e.proceeds.number - e.orig_basis.number for e in result.entries), Decimal(0)
    )
    assert total_reported_gl - total_open_basis_adj == total_economic_sold_gl


# ---- Existing example ledgers (test0 - test7) ----


@pytest.mark.parametrize(
    "filename,expected_total_gl",
    [
        ("wash_sales.test0.beancount", Decimal("-1000")),
        ("wash_sales.test1.beancount", Decimal("0")),
        ("wash_sales.test2.beancount", Decimal("-100")),
        ("wash_sales.test3.beancount", Decimal("-4200")),
        ("wash_sales.test4.beancount", Decimal("-4200")),
        ("wash_sales.test5.beancount", Decimal("100")),
        ("wash_sales.test6.beancount", Decimal("-12000")),
        ("wash_sales.test7.beancount", Decimal("100")),
    ],
)
def test_all_examples_conserve_money(filename: str, expected_total_gl: Decimal):
    entries = _load_example(filename)
    for unadj in (False, True):
        res = ws.calculate_wash_sales(entries, "STONK", unadjusted_basis_code_b=unadj)
        _assert_conservation_invariant(res)
        assert len(res.open_lots) == 0
        total_gl = sum((e.gl.number for e in res.entries), Decimal(0))
        assert total_gl == expected_total_gl


def test_example0_partial_lot_splitting():
    res = ws.calculate_wash_sales(_load_example("wash_sales.test0.beancount"), "STONK")
    _assert_conservation_invariant(res)
    assert len(res.entries) == 4
    # Jan 3: 5 sh from Jan 1 lot washed into 5 of 10 shares of Jan 2 lot
    assert res.entries[0].code_w is True
    assert res.entries[0].disallowed_loss.number == Decimal("250")
    assert res.entries[0].gl.number == Decimal("0")
    # Jan 4: sells the 5 washed shares of Jan 2 lot (basis 500 + 250 = 750 -> loss 500 washed into remaining 5 of Jan 1)
    assert res.entries[1].code_w is True
    assert res.entries[1].basis.number == Decimal("750")
    assert res.entries[1].disallowed_loss.number == Decimal("500")
    assert res.entries[1].gl.number == Decimal("0")
    # Jan 5: sells remaining 5 of Jan 1 (basis 500 + 500 = 1000 -> -750) and 5 unwashed of Jan 2 (basis 500 -> -250)
    assert [e.gl.number for e in res.entries[2:]] == [Decimal("-750"), Decimal("-250")]


def test_example1_per_lot_fifo_matching_not_averaged():
    """Treas. Reg. § 1.1091-1(c): match replacement shares to earliest-acquired loss lot first."""
    res = ws.calculate_wash_sales(_load_example("wash_sales.test1.beancount"), "STONK")
    _assert_conservation_invariant(res)
    assert len(res.entries) == 3

    # Lot 1: 10 sh acquired 2020-12-25 @ 100, sold @ 80 (-20/sh); 8 sh washed -> 160 disallowed, -40 realized
    e0 = res.entries[0]
    assert e0.property.number == Decimal("10")
    assert e0.orig_acquired == date(2020, 12, 25)
    assert e0.code_w is True
    assert e0.disallowed_loss.number == Decimal("160")
    assert e0.gl.number == Decimal("-40")
    assert e0.is_long_term is True

    # Lot 2: 5 sh acquired 2021-01-25 @ 120, sold @ 80 (-40/sh); no replacement left -> -200 realized
    e1 = res.entries[1]
    assert e1.property.number == Decimal("5")
    assert e1.orig_acquired == date(2021, 1, 25)
    assert e1.code_w is False
    assert e1.gl.number == Decimal("-200")
    assert e1.is_long_term is True

    # Lot 3: 8 sh @ 50 + 20/sh wash adj = 560 basis, sold @ 100 = 800 proceeds -> +240 Long-Term (tacked!)
    e2 = res.entries[2]
    assert e2.property.number == Decimal("8")
    assert e2.basis.number == Decimal("560")
    assert e2.gl.number == Decimal("240")
    assert e2.is_long_term is True


def test_example6_10b5_monthly_vests_2024_leap_year():
    res = ws.calculate_wash_sales(_load_example("wash_sales.test6.beancount"), "STONK")
    _assert_conservation_invariant(res)
    # 2024 calendar gaps between 25ths:
    # Jan->Feb 31d (no), Feb->Mar 29d (W), Mar->Apr 31d (no, -2000), Apr->May 30d (W),
    # May->Jun 31d (-2000), Jun->Jul 30d (W), Jul->Aug 31d (-2000), Aug->Sep 31d (-1000),
    # Sep->Oct 30d (W), Oct->Nov 31d (-2000), Nov->Dec 30d (W), Dec (-2000)
    expected_gls = [
        Decimal("-1000"),
        Decimal("0"),
        Decimal("-2000"),
        Decimal("0"),
        Decimal("-2000"),
        Decimal("0"),
        Decimal("-2000"),
        Decimal("-1000"),
        Decimal("0"),
        Decimal("-2000"),
        Decimal("0"),
        Decimal("-2000"),
    ]
    assert [e.gl.number for e in res.entries] == expected_gls


def test_example7_augmented_same_cost_lots_and_trade_date():
    """Fixes the old $160 phantom-basis bug on augmented lots + checks trade-date override."""
    res = ws.calculate_wash_sales(_load_example("wash_sales.test7.beancount"), "STONK")
    _assert_conservation_invariant(res)
    # Sold 10 sh (2020-12-25) -> fully washed across the 8+2 augmented lots (200 disallowed, 0 G/L)
    # Sold 5 sh (2021-01-25) -> unwashed (-200 G/L)
    # Sold 8 sh + 2 sh on trade-date 2023-01-04 -> basis 560 + 140 = 700, proceeds 1000 -> +300 G/L
    assert res.entries[0].gl.number == Decimal("0")
    assert res.entries[0].disallowed_loss.number == Decimal("200")
    assert res.entries[1].gl.number == Decimal("-200")
    sold_on_override = [e for e in res.entries if e.sold == date(2023, 1, 4)]
    assert len(sold_on_override) == 2
    assert sum((e.basis.number for e in sold_on_override), Decimal(0)) == Decimal("700")
    assert sum((e.gl.number for e in sold_on_override), Decimal(0)) == Decimal("300")


# ---- Targeted IRS IRC § 1091 & § 1223(3) edge cases ----


def test_replacement_ordered_by_date_not_price():
    """Regression test: Cost tuple had price before date; must match earliest DATE first."""
    entries = _load_str(
        """
        2022-01-01 * "Old lot"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {100 USD, 2022-01-01}

        2023-01-05 * "Older but MORE EXPENSIVE replacement candidate"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {500 USD, 2023-01-05}

        2023-01-20 * "Newer but CHEAPER replacement candidate"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {50 USD, 2023-01-20}

        2023-01-10 * "Loss sale of old lot"
            Assets:Brokerage:STONK  -10 STONK {2022-01-01} @ 80 USD
            Assets:Checking         800 USD
            Income:PnL
        """
    )
    res = ws.calculate_wash_sales(entries, "STONK")
    _assert_conservation_invariant(res)
    assert len(res.open_lots) == 2
    jan05_lot = next(l for l in res.open_lots if l.acquired_date == date(2023, 1, 5))
    jan20_lot = next(l for l in res.open_lots if l.acquired_date == date(2023, 1, 20))
    # Jan 5 (earlier date) MUST receive the $20/sh wash adjustment, NOT Jan 20 ($50 cheaper price)
    assert jan05_lot.is_replacement is True
    assert jan05_lot.basis_adj_per_share == Decimal("20")
    assert jan20_lot.is_replacement is False
    assert jan20_lot.basis_adj_per_share == Decimal("0")


def test_labeled_and_unlabeled_lots_same_price_and_date():
    """Regression test: mixing label=None and label='grantB' must not raise TypeError."""
    entries = _load_str(
        """
        2022-01-01 * "Old lot"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {100 USD, 2022-01-01}

        2023-01-10 * "Loss sale"
            Assets:Brokerage:STONK  -10 STONK {2022-01-01} @ 80 USD
            Assets:Checking         800 USD
            Income:PnL

        2023-01-15 * "Vest A (unlabeled)"
            Assets:Checking
            Assets:Brokerage:STONK    5 STONK {50 USD, 2023-01-15}

        2023-01-15 * "Vest B (labeled)"
            Assets:Checking
            Assets:Brokerage:STONK    5 STONK {50 USD, 2023-01-15, "grantB"}
        """
    )
    res = ws.calculate_wash_sales(entries, "STONK")
    _assert_conservation_invariant(res)
    assert res.entries[0].disallowed_loss.number == Decimal("200")
    assert all(l.is_replacement and l.basis_adj_per_share == Decimal("20") for l in res.open_lots)


def test_gain_lot_does_not_mask_loss_lot_in_same_sale():
    """A gain on Lot A in the same transaction must NOT cancel wash detection on Lot B."""
    entries = _load_str(
        """
        2022-01-01 * "Cheap lot"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {50 USD, 2022-01-01}

        2022-02-01 * "Expensive lot"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {150 USD, 2022-02-01}

        2023-06-01 * "Sell both @ 110 (net +200, but Lot 2 has -400 loss!)"
            Assets:Brokerage:STONK  -10 STONK {2022-01-01} @ 110 USD
            Assets:Brokerage:STONK  -10 STONK {2022-02-01} @ 110 USD
            Assets:Checking        2200 USD
            Income:PnL

        2023-06-15 * "RSU vest within 30 days"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {110 USD, 2023-06-15}
        """
    )
    res = ws.calculate_wash_sales(entries, "STONK")
    _assert_conservation_invariant(res)
    assert len(res.entries) == 2
    assert res.entries[0].gl.number == Decimal("600")
    assert res.entries[0].code_w is False
    assert res.entries[1].code_w is True
    assert res.entries[1].disallowed_loss.number == Decimal("400")
    assert res.entries[1].gl.number == Decimal("0")
    assert res.open_lots[0].adjusted_unit_cost == Decimal("150")


def test_holding_period_tacking_turns_short_term_into_long_term():
    """IRC § 1223(3): 350 days held + 30 days on replacement = 380 days -> Long-Term."""
    entries = _load_str(
        """
        2022-01-01 * "Initial vest"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {100 USD, 2022-01-01}

        2022-12-17 * "Sell at loss after 350 days (Short-Term)"
            Assets:Brokerage:STONK  -10 STONK {2022-01-01} @ 80 USD
            Assets:Checking         800 USD
            Income:PnL

        2022-12-27 * "Replacement vest (10 days later)"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {80 USD, 2022-12-27}

        2023-01-26 * "Sell replacement 30 days later (350+30=380 days -> Long-Term!)"
            Assets:Brokerage:STONK  -10 STONK {2022-12-27} @ 120 USD
            Assets:Checking        1200 USD
            Income:PnL
        """
    )
    res = ws.calculate_wash_sales(entries, "STONK")
    _assert_conservation_invariant(res)
    assert res.entries[0].is_long_term is False
    assert res.entries[0].code_w is True
    assert res.entries[1].is_long_term is True
    # 2022-12-27 - 350 days = 2022-01-11
    assert res.entries[1].acquired == date(2022, 1, 11)
    assert res.entries[1].basis.number == Decimal("1000")
    assert res.entries[1].gl.number == Decimal("200")


def test_long_term_exact_one_year_boundary():
    assert ws.is_long_term(date(2023, 1, 1), date(2024, 1, 1)) is False
    assert ws.is_long_term(date(2023, 1, 1), date(2024, 1, 2)) is True
    assert ws.is_long_term(date(2024, 2, 29), date(2025, 2, 28)) is False
    assert ws.is_long_term(date(2024, 2, 29), date(2025, 3, 1)) is True


def test_rsu_sell_to_cover_does_not_self_wash():
    """Selling 4 of 10 shares from a single RSU vest at a loss must NOT wash against the remaining 6."""
    entries = _load_str(
        """
        2024-03-15 * "RSU vest"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {100 USD, 2024-03-15}

        2024-03-15 * "Sell-to-cover 4 shares"
            Assets:Brokerage:STONK   -4 STONK {2024-03-15} @ 98 USD
            Assets:Checking         392 USD
            Income:PnL
        """
    )
    res = ws.calculate_wash_sales(entries, "STONK")
    _assert_conservation_invariant(res)
    assert len(res.entries) == 1
    assert res.entries[0].code_w is False
    assert res.entries[0].gl.number == Decimal("-8")
    assert len(res.open_lots) == 1
    assert res.open_lots[0].shares == Decimal("6")
    assert res.open_lots[0].basis_adj_per_share == Decimal("0")


def test_30_day_window_exact_boundaries_before_and_after():
    entries = _load_str(
        """
        2022-01-01 * "Old lot"
            Assets:Checking
            Assets:Brokerage:STONK   40 STONK {100 USD, 2022-01-01}

        2024-03-01 * "31 days BEFORE Apr 1 -> outside window"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {90 USD, 2024-03-01}

        2024-03-02 * "30 days BEFORE Apr 1 -> INSIDE window"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {90 USD, 2024-03-02}

        2024-04-01 * "Loss sale of 40 shares"
            Assets:Brokerage:STONK  -40 STONK {2022-01-01} @ 80 USD
            Assets:Checking        3200 USD
            Income:PnL

        2024-05-01 * "30 days AFTER Apr 1 -> INSIDE window"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {90 USD, 2024-05-01}

        2024-05-02 * "31 days AFTER Apr 1 -> outside window"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {90 USD, 2024-05-02}
        """
    )
    res = ws.calculate_wash_sales(entries, "STONK")
    _assert_conservation_invariant(res)
    # Exactly 20 of 40 shares washed (Mar 2 and May 1); Mar 1 and May 2 untouched
    assert res.entries[0].disallowed_loss.number == Decimal("400")
    assert res.entries[0].gl.number == Decimal("-400")
    replaced_dates = {l.acquired_date for l in res.open_lots if l.is_replacement}
    assert replaced_dates == {date(2024, 3, 2), date(2024, 5, 1)}


def test_partial_wash_then_sell_washed_sublot_leaves_unwashed_available():
    """Regression test for the old inv_wash desync bug."""
    entries = _load_str(
        """
        2022-01-01 * "Old lot 1"
            Assets:Checking
            Assets:Brokerage:STONK    6 STONK {100 USD, 2022-01-01}

        2022-02-01 * "Old lot 2"
            Assets:Checking
            Assets:Brokerage:STONK    4 STONK {100 USD, 2022-02-01}

        2023-05-01 * "New 10-share vest"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {80 USD, 2023-05-01}

        2023-05-05 * "Loss sale 1 (6 sh) -> washes 6 of the 10-sh vest"
            Assets:Brokerage:STONK   -6 STONK {2022-01-01} @ 70 USD
            Assets:Checking         420 USD
            Income:PnL

        2023-05-10 * "Sell 6 sh of the May 1 vest at a gain -> closes the 6 washed shares"
            Assets:Brokerage:STONK   -6 STONK {2023-05-01} @ 150 USD
            Assets:Checking         900 USD
            Income:PnL

        2023-05-15 * "Loss sale 2 (4 sh) -> MUST wash into the remaining 4 unwashed May 1 shares"
            Assets:Brokerage:STONK   -4 STONK {2022-02-01} @ 70 USD
            Assets:Checking         280 USD
            Income:PnL
        """
    )
    res = ws.calculate_wash_sales(entries, "STONK")
    _assert_conservation_invariant(res)
    assert res.entries[0].disallowed_loss.number == Decimal("180")
    assert res.entries[1].gl.number == Decimal("240")  # 900 - (480 + 180)
    assert res.entries[2].code_w is True
    assert res.entries[2].disallowed_loss.number == Decimal("120")
    assert res.entries[2].gl.number == Decimal("0")
    assert len(res.open_lots) == 1
    assert res.open_lots[0].shares == Decimal("4")
    assert res.open_lots[0].adjusted_unit_cost == Decimal("110")


def test_same_lot_partial_fills_at_different_prices():
    """Two postings from the same Cost lot at different sale prices must not overwrite each other."""
    entries = _load_str(
        """
        2023-01-01 * "Vest"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {100 USD, 2023-01-01}

        2023-06-01 * "Sell in two fills"
            Assets:Brokerage:STONK   -4 STONK {2023-01-01} @ 110 USD
            Assets:Brokerage:STONK   -6 STONK {2023-01-01} @ 120 USD
            Assets:Checking        1160 USD
            Income:PnL
        """
    )
    res = ws.calculate_wash_sales(entries, "STONK")
    _assert_conservation_invariant(res)
    assert sum((e.proceeds.number for e in res.entries), Decimal(0)) == Decimal("1160")
    assert sum((e.gl.number for e in res.entries), Decimal(0)) == Decimal("160")


def test_cross_year_wash_sale_and_year_filter():
    """Dec 20 loss washed by Jan 10 vest in the next year; --year filters output without breaking the window."""
    entries = _load_str(
        """
        2023-01-01 * "Old vest"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {100 USD, 2023-01-01}

        2024-12-20 * "Year-end loss sale"
            Assets:Brokerage:STONK  -10 STONK {2023-01-01} @ 70 USD
            Assets:Checking         700 USD
            Income:PnL

        2025-01-10 * "January vest within 30 days"
            Assets:Checking
            Assets:Brokerage:STONK   10 STONK {75 USD, 2025-01-10}

        2025-03-01 * "Sell January vest"
            Assets:Brokerage:STONK  -10 STONK {2025-01-10} @ 90 USD
            Assets:Checking         900 USD
            Income:PnL
        """
    )
    e_2024 = ws.calculate_8949_entries(entries, "STONK", year=2024)
    e_2025 = ws.calculate_8949_entries(entries, "STONK", year=2025)
    assert len(e_2024) == 1
    assert e_2024[0].code_w is True
    assert e_2024[0].disallowed_loss.number == Decimal("300")
    assert e_2024[0].gl.number == Decimal("0")

    assert len(e_2025) == 1
    assert e_2025[0].basis.number == Decimal("1050")
    assert e_2025[0].gl.number == Decimal("-150")
    assert e_2025[0].is_long_term is True  # Tacked > 1 year from 2023-01-01!


def test_in_kind_account_transfer_is_ignored():
    entries = _load_str(
        """
        2023-01-01 * "Vest"
            Assets:Checking
            Assets:Brokerage:STONK    10 STONK {100 USD, 2023-01-01}

        2023-06-01 * "ACATS transfer"
            Assets:Brokerage:STONK   -10 STONK {100 USD, 2023-01-01}
            Assets:Brokerage2:STONK   10 STONK {100 USD, 2023-01-01}

        2024-06-01 * "Sell after transfer"
            Assets:Brokerage2:STONK  -10 STONK {100 USD, 2023-01-01} @ 120 USD
            Assets:Checking         1200 USD
            Income:PnL
        """
    )
    res = ws.calculate_wash_sales(entries, "STONK")
    _assert_conservation_invariant(res)
    assert len(res.entries) == 1
    assert res.entries[0].gl.number == Decimal("200")
    assert res.entries[0].is_long_term is True


def test_unadjusted_basis_code_b_mode():
    entries = _load_example("wash_sales.test3.beancount")
    res_std = ws.calculate_wash_sales(entries, "STONK", unadjusted_basis_code_b=False)
    res_b = ws.calculate_wash_sales(entries, "STONK", unadjusted_basis_code_b=True)
    _assert_conservation_invariant(res_std)
    _assert_conservation_invariant(res_b)

    # Second sale (10 sh @ 90, orig basis 5000, prior wash adj +100 -> true loss -4200):
    assert res_std.entries[1].basis.number == Decimal("5100")
    assert res_std.entries[1].code_b is False
    assert res_std.entries[1].gl_adjustment.number == Decimal("0")
    assert res_std.entries[1].gl.number == Decimal("-4200")

    assert res_b.entries[1].basis.number == Decimal("5000")
    assert res_b.entries[1].code_b is True
    assert res_b.entries[1].gl_adjustment.number == Decimal("-100")
    assert res_b.entries[1].gl.number == Decimal("-4200")
