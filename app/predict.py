"""Command-line inference using the same engine as the local browser app."""
import argparse
import json
from pathlib import Path
from engine import Predictor


def main():
    parser=argparse.ArgumentParser(description='Frozen-model impact force history prediction')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input',type=Path,help='JSON with name and inputs')
    group.add_argument('--batch',type=Path,help='CSV using the bundled template')
    group.add_argument('--template',action='store_true',help='Write batch template without inference')
    parser.add_argument('--output',type=Path,required=True,help='Output directory')
    args=parser.parse_args()
    predictor=Predictor()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.template:
        (args.output/'batch_template.csv').write_text(predictor.template_csv(),encoding='utf-8-sig')
    elif args.batch:
        result=predictor.batch_csv(args.batch.read_text(encoding='utf-8-sig'))
        (args.output/'anchors.csv').write_text(result['summary_csv'],encoding='utf-8-sig')
        (args.output/'force_histories.csv').write_text(result['curves_csv'],encoding='utf-8-sig')
        for r in result['results']:r.pop('curve_csv',None)
        (args.output/'predictions.json').write_text(json.dumps(result['results'],ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        result=predictor.predict(json.loads(args.input.read_text(encoding='utf-8-sig')))
        (args.output/'force_history.csv').write_text(result.pop('curve_csv'),encoding='utf-8-sig')
        (args.output/'prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(args.output.resolve()))


if __name__=='__main__':main()
