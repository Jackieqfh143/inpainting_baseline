from prettytable import from_csv
from prettytable import PrettyTable
import csv
import pandas as pd
import re

def is_number_or_float(sample_str):
    ''' Returns True if the string contains only
        number or float '''
    result = False
    if re.match(r"[-+]?(?:\d*\.\d+|\d+)", sample_str) is not None:
        result = True
    return result

def pretty_table(save_file,title,sortby):
    df = pd.read_csv(save_file)
    df.sort_values(sortby, ascending=False, inplace=True)
    df.to_csv(save_file, index=False)

    table = PrettyTable()
    with open(save_file, "r") as fp:
        x = from_csv(fp)
        table.title = title
        table.field_names = x.field_names
        for i, row in enumerate(x.rows):
            new_row = []
            for j, item in enumerate(row):
                if is_number_or_float(item) == True:
                    new_row.append(f'{float(item):.4f}')
                else:
                    new_row.append(item)
            table.add_row(new_row)

        print(table)

    with open(save_file, 'w') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(table.field_names)
        writer.writerows(table.rows)
        print('done recording!')

if __name__ == '__main__':
    save_file = './paris_results(30%~40%).csv'
    sortby = 'FID'
    pretty_table(save_file,sortby)
