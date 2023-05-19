#!/usr/bin/env python3
"""Software for managing and analysing patients' inflammation data in our imaginary hospital."""

import argparse
from inflammation import models, views


def main(args):
    """The MVC Controller of the patient inflammation data system.

    The Controller is responsible for:
    - selecting the necessary models and views for the current task
    - passing data between models and views
    """
    # read in in_files arg
    in_files = args.infiles

    # convert to list
    if not isinstance(in_files, list):
        in_files = [args.infiles]

    # iterate through in_files
    for file_name in in_files:
        # load inflammation data
        inflammation_data = models.load_csv(file_name)

        # get statistics
        view_data = {
            'average': models.daily_mean(inflammation_data),
            'max': models.daily_max(inflammation_data),
            'min': models.daily_min(inflammation_data)
        }

        # visualise the data
        views.visualize(view_data)


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description='A basic patient inflammation data management system'
    )

    # add in_files argument
    parser.add_argument(
        'in_files',
        nargs='+',
        help='Input CSV(s) containing inflammation series for each patient'
    )

    # parse arguments
    args = parser.parse_args()

    # run main function
    main(args)
