from .OFTime import OFTimestring

def OFKey(id, field, time=None, rank=None, index=None):
    """Build a radex store key, mirroring Foam::radexKey.

    Components are appended in the order the C++ side uses:
    ``<id>_<field>[_<time>][_<rank>][_<index>]``. Any may be omitted,
    including ``id``, to address a key written without that component.
    """
    key = f"{id}_{field}" if id is not None else f"{field}"
    if time is not None:
        key += f"_{OFTimestring(time)}"
    if rank is not None:
        key += f"_{rank}"
    if index is not None:
        key += f"_{index}"
    return key