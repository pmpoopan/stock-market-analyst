"""Extended mock OHLCV bars for technical indicator tests."""

from datetime import date, timedelta

from app.models.schemas import OHLCVBar


def make_trending_bars(
    count: int = 250,
    start_price: float = 1000.0,
    daily_change: float = 2.0,
    start_date: date = date(2025, 1, 1),
) -> list[OHLCVBar]:
    """Generate a steadily rising OHLCV series for indicator testing."""
    bars: list[OHLCVBar] = []
    for i in range(count):
        close = start_price + i * daily_change
        bars.append(
            OHLCVBar(
                date=start_date + timedelta(days=i),
                open=round(close - 1.5, 4),
                high=round(close + 3.0, 4),
                low=round(close - 4.0, 4),
                close=round(close, 4),
                volume=1_000_000 + (i % 10) * 25_000,
            )
        )
    return bars


def make_flat_bars(count: int = 30, price: float = 100.0) -> list[OHLCVBar]:
    """Constant-price bars — SMA/EMA should equal price once seeded."""
    return [
        OHLCVBar(
            date=date(2026, 1, 1) + timedelta(days=i),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=500_000,
        )
        for i in range(count)
    ]


def make_volatile_bars(count: int = 50, start_price: float = 500.0) -> list[OHLCVBar]:
    """Alternating up/down bars for RSI and OBV tests."""
    bars: list[OHLCVBar] = []
    close = start_price
    for i in range(count):
        direction = 1 if i % 2 == 0 else -1
        close = round(close + direction * 5, 4)
        bars.append(
            OHLCVBar(
                date=date(2026, 1, 1) + timedelta(days=i),
                open=round(close - direction, 4),
                high=round(close + 3, 4),
                low=round(close - 3, 4),
                close=close,
                volume=800_000 + direction * 50_000,
            )
        )
    return bars
