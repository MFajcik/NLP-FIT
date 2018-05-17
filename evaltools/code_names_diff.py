# Martin Fajcik BUT@FIT 2018
import argparse

import os

import pandas as pd
import sys
from tqdm import tqdm
from colour import Color

from other.logging_config import init_logging
import re

'''Globals'''
# Evaluation result file column indices
HINT_COLUMN = 0
EXPECTED_COLUMN = 1
LIST_COLUMN = 2
PRECISION_COLUMN = 3

# Color gradient of diffs
COLOR_GRADIENT_FROM = "red"
COLOR_GRADIENT_TO = "blue"

logging = None
trailing_string = 'Total average precision:'

# Maximum length of items in list column
FRAME_LIST_LENGTH = 30

# Length of one result in vertical_axis (RESULT_STEP rows)
RESULT_STEP = 3


def drop_non_relevant(df):
    rows = list(df.iterrows())
    for i, row in rows:
        if row[0].startswith(trailing_string):
            total_precision = float(row[0].split(":")[1])
            oov = float(df.iloc[i + 1][0].split(":")[1])
            df = df.drop(range(i, len(df)))
            return df, total_precision, oov
    logging.critical('End of evaluation file not found!')
    raise ValueError('End of evaluation file not found!')


IDX_COL = 0
AP_COL = 1
DIFF_COL = 2
DIFF_ORDER_COL = 3

rgx_rm_brackets = re.compile(r'\(.*?\)')
rm_brackets = lambda s: rgx_rm_brackets.sub("", s)


def diff_results(df_result_a, df_result_b, sort):
    """

    :param df_result_a:
    :param df_result_b:
    :param sort:
    :return:
    """
    # contains_word_counts_a = check_if_contains_word_counts(df_result_a)
    # contains_word_counts_b = check_if_contains_word_counts(df_result_b)
    wordstep_a = 2
    wordstep_b = 2
    diff_result = list()
    df_result_a, total_avg_precision_a, oov_a = drop_non_relevant(df_result_a)
    df_result_b, total_avg_precision_b, oov_b = drop_non_relevant(df_result_b)
    for index_a, row_a in tqdm(list(df_result_a.iterrows())):
        change_flag = False
        diff_order = list()
        has_pair = False
        row_b = None
        row_a_OOV = row_a[LIST_COLUMN] == "Hint is out of vocabulary"
        # remove everything without brackets
        # because word counts, which are in brackets if present, do not have to match
        for _, row_b in df_result_b.iterrows():
            row_b_OOV = row_b[LIST_COLUMN] == "Hint is out of vocabulary"
            if row_a_OOV or row_b_OOV:
                has_pair = True
                # TODO: Handle OOV tests
                pass
            elif rm_brackets(row_a[HINT_COLUMN]) == rm_brackets(row_b[HINT_COLUMN]) and \
                    row_a[EXPECTED_COLUMN] == row_b[EXPECTED_COLUMN]:
                tokens_a = re.split(':|\s', row_a[LIST_COLUMN].strip())[::wordstep_a]
                tokens_b = re.split(':|\s', row_b[LIST_COLUMN].strip())[::wordstep_b]
                nobr_tokens_a = list(map(rm_brackets, tokens_a))
                nobr_tokens_b = list(map(rm_brackets, tokens_b))
                s_tokens_a = set(nobr_tokens_a)
                s_tokens_b = set(nobr_tokens_b)
                # this solves OOV words not present in both sets
                if s_tokens_a.issubset(s_tokens_b) or s_tokens_b.issubset(s_tokens_a):
                    has_pair = True
                    order_dict = dict(zip(nobr_tokens_a, range(len(nobr_tokens_a))))
                    for token in nobr_tokens_b:
                        diff_order.append(order_dict.get(token, len(diff_order)))
                    # In case some token of b was not between tokens of a, we need to put all numbers lin
                    missing = []
                    for x in range(len(diff_order)):
                        if x not in diff_order:
                            missing.append(x)
                    diff_order += missing

                    for i, x in enumerate(diff_order):
                        if i != x:
                            change_flag = True
                            break
                    break
        if not has_pair:
            logging.critical("CONSISTENCY ERROR: Row {} of input1 file does not have a pair!".format(index_a))
            exit(1)
        if change_flag:
            ap_a = float(row_a[PRECISION_COLUMN].split(':')[-1])
            ap_b = float(row_b[PRECISION_COLUMN].split(':')[-1])
            avg_prec_diff = ap_b - ap_a
            list_a = row_a[LIST_COLUMN].split()
            list_b = row_b[LIST_COLUMN].split()
            list_tuples = list()
            for i in range(len(list_a)):
                items_a = list_a[i].split(":")
                if i < len(list_b):
                    items_b = list_b[i].split(":")
                else:
                    items_b = ("", "")

                class Diff:
                    def __init__(self, hint=None, expected=None, list=None):
                        self.hint = hint
                        self.expected = expected
                        self.list = list

                class DiffItem:
                    def __init__(self, name=None, occurences=None, distance_from_hint=None):
                        self.name = name
                        self.occurences = occurences
                        self.distance_from_hint = distance_from_hint

                # if contains_word_counts_a:
                #     diff_item_a = DiffItem(items_a[0],items_a[1],items_a[2])
                # else:
                diff_item_a = DiffItem(items_a[0], None, items_a[1])
                # if contains_word_counts_b:
                #     diff_item_b = DiffItem(items_b[0],items_b[1],items_b[2])
                # else:
                diff_item_b = DiffItem(items_b[0], None, items_b[1])
                list_tuples.append((diff_item_a, diff_item_b))

            diff_result.append((index_a, avg_prec_diff,
                                Diff(row_a[HINT_COLUMN],
                                     row_a[EXPECTED_COLUMN], list_tuples),
                                diff_order))
    total_avg_prec_diff = total_avg_precision_b - total_avg_precision_a
    total_oov_diff = oov_b - oov_a
    sorted_diffs = sorted(diff_result, key=lambda x: x[1], reverse=True) if sort else diff_result
    return total_avg_prec_diff, total_oov_diff, sorted_diffs


V_OFFSET_START = 4  # dataframe index, at which ordered list start
H_OFFSET_START = 1

def add_formatting_xslx(df, diffs, writer):
    """

    :param df:
    :param diffs:
    :param writer:
    :return:
    """
    wb = writer.book
    ws = writer.sheets['Diff']
    ws.set_column('E:BA', 15)
    ws.set_column('C:D', 25)
    h_offset = H_OFFSET_START
    for diff in tqdm(diffs):
        expected = list(filter(None, re.split('/|\s', diff[DIFF_COL].expected)))
        diff_order = diff[DIFF_ORDER_COL]
        colors = list(Color(COLOR_GRADIENT_FROM).range_to(Color(COLOR_GRADIENT_TO), len(diff_order)))
        default_order = range(len(diff_order))
        v_offset = V_OFFSET_START
        for color_idx_a, color_idx_b in zip(default_order, diff_order):
            set_cell_color(wb, ws, df, h_offset, v_offset, colors[color_idx_a].get_hex_l(), expected)
            set_cell_color(wb, ws, df, h_offset + 1, v_offset, colors[color_idx_b].get_hex_l(), expected)
            v_offset += 1
        h_offset += RESULT_STEP

    return writer


def set_cell_color(wb, ws, df, row, col, color, expected):
    """

    :param wb:
    :param ws:
    :param df:
    :param row:
    :param col:
    :param color:
    :param expected:
    """
    content = df.iloc[row - H_OFFSET_START][col]
    opts = {'bold': rm_brackets(content) in expected, 'bg_color': color}
    myformat = wb.add_format(opts)
    ws.write(row, col, content, myformat)


def align(list_to_align, length=FRAME_LIST_LENGTH):
    """

    :param list_to_align:
    :param length:
    :return:
    """
    togen = length - len(list_to_align)
    return list_to_align + ["_" for _ in range(togen)]


def form_data_frame(t_avp, t_oov, diffs):
    """

    :param t_avp:
    :param t_oov:
    :param diffs:
    :return:
    """
    df = pd.DataFrame(columns=align(["Index", "AVP Diff", "Hint", "Explanation"]))
    idx = 0
    for diff in tqdm(diffs):
        tmp1 = [diff[IDX_COL],
                diff[AP_COL], diff[DIFF_COL].hint, diff[DIFF_COL].expected]
        tmp2 = [""] + [diff[AP_COL]] + [""] * (len(tmp1) - 2)
        for ditem1, ditem2 in diff[DIFF_COL].list:
            tmp1.append(ditem1.name)
            tmp2.append(ditem2.name)
        df.loc[idx] = align(tmp1)
        df.loc[idx + 1] = align(tmp2)
        df.loc[idx + 2] = align([])
        idx += RESULT_STEP
    df.loc[idx] = align(["Average Precision Diff", t_avp])
    df.loc[idx + 1] = align(["Out of Vocabulary Diff", t_oov])
    return df


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--input1",
                        required=True,
                        help="Input file in cnwae format.")
    parser.add_argument("-i2", "--input2",
                        required=True,
                        help="Input file in cnwae format.")
    parser.add_argument("-o", "--output",
                        required=True,
                        help="Output file.")
    parser.add_argument("-l", "--logpath",
                        default=os.getcwd(),
                        help="Explicit setting of log folder path")
    parser.add_argument("-s", "--sorted",
                        action='store_true',
                        default=False,
                        help="Sorts diff according to average MAP differences")
    args = parser.parse_args(args)
    logging = init_logging(os.path.basename(sys.argv[0]).split(".")[0], logpath=args.logpath)
    print("Reading tsvs...")
    result1 = pd.read_csv(args.input1, delimiter="\t", header=None)
    result2 = pd.read_csv(args.input2, delimiter="\t", header=None)
    print("Comparing list orders...")
    t_avp_diff, t_oov_diff, sequence_diff = diff_results(result1, result2, args.sorted)
    print("Forming data frame...")
    df = form_data_frame(t_avp_diff, t_oov_diff, sequence_diff)
    print("Formatting...")
    writer = pd.ExcelWriter("{}.xlsx".format(args.output), engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Diff', index=None)
    writer = add_formatting_xslx(df, sequence_diff, writer)
    print("Saving diff to {}.".format("{}.xlsx".format(args.output)))
    writer.save()


if __name__ == "__main__":
    main(sys.argv[1:])
