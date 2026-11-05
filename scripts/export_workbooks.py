"""Export inspectable Excel copies of frozen numeric source tables using openpyxl."""
from pathlib import Path
import argparse
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

ROOT = Path(__file__).resolve().parents[1]


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    experiment = 'reports' if (ROOT/'reports/Table_5.csv').exists() else 'results/experiment'
    books = {
        'Dataset_and_splits.xlsx': [('Archive','data/archive_199.csv'),('Input_dictionary','data/input_dictionary.csv'),
             ('Sources','data/source_mapping.csv'),('Own_5000_points','data/present10_origin_long.csv')]
             +[(str(s),f'data/splits/{s}.csv') for s in (20261102,20261103,20261104)],
        'Prediction_results.xlsx': [('Table5',experiment+'/Table_5.csv'),('Table6b',experiment+'/Table_6b.csv'),
             ('Production_Table5','results/production/Table_5.csv'),('Production_Table6b','results/production/Table_6b.csv'),
             ('Negative_cases',experiment+'/negative_R2_cases.csv'),('Source_bootstrap',experiment+'/paired_source_bootstrap.csv'),
             ('SHAP','results/production/shap_values.csv')],
    }
    for name, tables in books.items():
        book = Workbook(); book.remove(book.active)
        book.properties.creator = 'fffaang'; book.properties.lastModifiedBy = 'fffaang'
        for sheet_name, relative in tables:
            if not (ROOT/relative).exists(): continue
            frame = pd.read_csv(ROOT/relative)
            ws = book.create_sheet(sheet_name)
            ws.append(frame.columns.tolist())
            for row in frame.itertuples(index=False, name=None):
                ws.append([None if pd.isna(v) else v for v in row])
            for cell in ws[1]: cell.font = Font(bold=True)
            ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
            for cells in ws.columns:
                width = min(40, max(12, max(len(str(c.value or '')) for c in list(cells)[:40])+2))
                ws.column_dimensions[cells[0].column_letter].width = width
        book.save(output/name)
        print(name, len(book.sheetnames), 'sheets')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'output/workbooks')
    export(p.parse_args().output)
