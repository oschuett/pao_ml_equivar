# -*- coding: utf-8 -*-

import re

#===============================================================================
def read_energy(fn):
    try:
        content = open(fn).read()
        m = re.search("ENERGY\|(.*)", content)
        return float(m.group(1).split()[-1])
    except:
        print("error with: "+fn)
    return float("NaN")

#EOF
