#!/usr/bin/python3

from pprint import pprint

def main():
    kinds = {}
    fn = "2H2O_MD/frame_0000/2H2O_pao44-1_0.pao"
    for line in open(fn).readlines():
        parts = line.split()
        if parts[0] == "Parametrization":
            assert parts[1] == "DIRECT"

        elif parts[0] == "Kind":
            kinds[parts[1]] = {'name': parts[2], 'atomic_number': int(parts[3])}

        elif parts[0] == "NParams":
            kinds[parts[1]]['nparams'] = int(parts[2])

    pprint(kinds)
    
main()