import math


def get_size_label(size_in_bytes, decimal_places=2):

    units = ('','K','M','G','T','P','E','Z','Y')
    conversion_factor = 1024

    if size_in_bytes == 0:
        index, size = 0, 0
    else:
        index = int(math.floor(math.log(size_in_bytes, conversion_factor)))
        size = size_in_bytes / math.pow(conversion_factor, index)

    return f'{size:,.{decimal_places}f} {units[index]}B'