from decimal import Decimal

def OFTimestring(value):
    """Format an OpenFOAM-style simulation time as a filesystem-safe string.

    Examples:
        0 -> "0"
        0.0 -> "0"
        0.1 -> "0.1"
        0.0500 -> "0.05"
        100.000 -> "100"
    """
    num = Decimal(str(value))
    if num == num.to_integral_value():
        return str(num.quantize(Decimal("1")))

    s = format(num.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s
