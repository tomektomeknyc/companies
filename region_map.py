# region_map.py
SUFFIX_REGION = {
    ".NZ": "ASIA_PAC_EX_JPN",
    ".AU": "ASIA_PAC_EX_JPN",
    ".DE": "EUROPE",
    ".FR": "EUROPE",
    ".O" : "NORTH_AMERICA",
    ".N" : "NORTH_AMERICA",
    # extend as needed
}

def region_for_ticker(ticker: str) -> str | None:
    """
    Return FF-5 region string for a ticker like 'SKC.NZ'.
    Looks at everything *after* the first dot.
    """
    if "." not in ticker:
        return None
    suffix = ticker[ticker.find(".") :].upper()   # '.NZ'
    return SUFFIX_REGION.get(suffix)
