#!/usr/bin/env python

# Loading libraries
import numpy as np
import pandas as pd

# For interfacing with the file system
import fnmatch
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='FCSlib: A python based toolkit for analyzing and processing flow cytometry and qPCR data')
    subparser = parser.add_subparsers(title='subcommands', dest='subcommand')
    # parse commands related to analysis
    aparser = subparser.add_parser('flow', help='suboptions for fcs')
    aparser.add_argument('-i', dest='infile', nargs='+', type=str, help='input files')
    aparser.add_argument('-n', dest='n_samples', type=int, help='number of samples to keep')
    aparser.add_argument('-trim', dest='trim', action='store_true', help='trim data in fcs file')
    # parse arguments
    args = parser.parse_args()
    if args.subcommand=='flow':
        df.to_csv(args.ofile, index=False)

if __name__ == "__main__":
    main()
