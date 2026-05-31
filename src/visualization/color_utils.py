import matplotlib.colors as mcolors

def colorstr_to_bgr(color):
    """
    Returns bgr-tupel for color String or Hex-code. Needed for CV2 plots.

    Parameters
    -------
    color: str
        string (e.g. "yellow") or hex code (e.g. "#036ffc")

    Returns
    -------
    tuple[int,int,int]
        tuple with b, g, r value

    Raises
    -------
    ValueError
        If parameter is not a string.
    
    """
    if isinstance(color, str):
        r, g, b = mcolors.to_rgb(color)
    else:
        raise ValueError(f"Color format not supported: {color}. Expecting string for color name or hex code.")
    
    return (int(b*255), int(g*255), int(r*255))