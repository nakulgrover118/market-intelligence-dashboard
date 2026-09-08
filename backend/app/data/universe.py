"""The pilot instrument universe.

Kept as plain data (not fetched from anywhere dynamic) so the exact set of
instruments used in any given run of the pipeline is visible in version
control, not hidden behind a live API call.
"""

from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    STOCK = "stock"
    INDEX = "index"
    COMMODITY = "commodity"


@dataclass(frozen=True)
class Instrument:
    ticker: str  # Yahoo Finance ticker symbol
    name: str
    asset_class: AssetClass
    sector: str | None = None  # None for indices/commodities


STOCKS: list[Instrument] = [
    Instrument("TCS.NS", "Tata Consultancy Services", AssetClass.STOCK, "IT"),
    Instrument("INFY.NS", "Infosys", AssetClass.STOCK, "IT"),
    Instrument("HCLTECH.NS", "HCL Technologies", AssetClass.STOCK, "IT"),
    Instrument("WIPRO.NS", "Wipro", AssetClass.STOCK, "IT"),
    Instrument("HDFCBANK.NS", "HDFC Bank", AssetClass.STOCK, "Banking"),
    Instrument("ICICIBANK.NS", "ICICI Bank", AssetClass.STOCK, "Banking"),
    Instrument("SBIN.NS", "State Bank of India", AssetClass.STOCK, "Banking"),
    Instrument("KOTAKBANK.NS", "Kotak Mahindra Bank", AssetClass.STOCK, "Banking"),
    Instrument("AXISBANK.NS", "Axis Bank", AssetClass.STOCK, "Banking"),
    Instrument("RELIANCE.NS", "Reliance Industries", AssetClass.STOCK, "Energy"),
    Instrument("ONGC.NS", "Oil & Natural Gas Corporation", AssetClass.STOCK, "Energy"),
    Instrument("NTPC.NS", "NTPC", AssetClass.STOCK, "Energy"),
    Instrument("ITC.NS", "ITC", AssetClass.STOCK, "FMCG"),
    Instrument("HINDUNILVR.NS", "Hindustan Unilever", AssetClass.STOCK, "FMCG"),
    Instrument("NESTLEIND.NS", "Nestle India", AssetClass.STOCK, "FMCG"),
    Instrument("MARUTI.NS", "Maruti Suzuki", AssetClass.STOCK, "Auto"),
    # Tata Motors demerged 2025-10-01; passenger vehicles (incl. JLR) now trade as TMPV.
    # History before the demerger date is Tata Motors' pre-split combined-entity data.
    Instrument("TMPV.NS", "Tata Motors Passenger Vehicles", AssetClass.STOCK, "Auto"),
    Instrument("M&M.NS", "Mahindra & Mahindra", AssetClass.STOCK, "Auto"),
    Instrument("SUNPHARMA.NS", "Sun Pharmaceutical", AssetClass.STOCK, "Pharma"),
    Instrument("DRREDDY.NS", "Dr. Reddy's Laboratories", AssetClass.STOCK, "Pharma"),
    Instrument("LT.NS", "Larsen & Toubro", AssetClass.STOCK, "Infrastructure"),
    Instrument("TATASTEEL.NS", "Tata Steel", AssetClass.STOCK, "Metals"),
    Instrument("BHARTIARTL.NS", "Bharti Airtel", AssetClass.STOCK, "Telecom"),
]

INDICES: list[Instrument] = [
    Instrument("^NSEI", "Nifty 50", AssetClass.INDEX),
    Instrument("^NSEBANK", "Nifty Bank", AssetClass.INDEX),
]

COMMODITIES: list[Instrument] = [
    Instrument("GOLDBEES.NS", "Nippon India Gold ETF (gold proxy)", AssetClass.COMMODITY),
    Instrument("SILVERBEES.NS", "Nippon India Silver ETF (silver proxy)", AssetClass.COMMODITY),
]

UNIVERSE: list[Instrument] = STOCKS + INDICES + COMMODITIES
