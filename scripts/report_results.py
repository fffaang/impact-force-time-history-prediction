from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]


def report():
    path=ROOT/'reports/acceptance.json'
    lines=['# 8:1:1 optimization evidence','',
        'Three predeclared partitions are reported separately. They are archive-internal reevaluations, not blind or unseen-source tests.',
        'The ten present-study histories no longer form an independent external validation set.','']
    if not path.exists():lines+=['## Status','','Formal validation/test evaluation has not completed. No new accuracy claim is available.','']
    else:
        data=json.loads(path.read_text());lines+=['## Acceptance','','| Split seed | Arm | Status | Anchor targets / 10 | Validation curve median R2 | Test curve median R2 | Negative val/test |',
            '|---|---|---|---:|---:|---:|---|']
        for r in data.get('acceptance',[]):
            if r['status']=='NOT_COMPLETED':lines.append(f'| {r["split_seed"]} | {r["arm"]} | NOT COMPLETED | | | | |');continue
            lines.append(f'| {r["split_seed"]} | {r["arm"]} | {r["status"]} | {r["anchor_targets_passed"]} | {r["validation_curve_median_R2"]:.4f} | {r["test_curve_median_R2"]:.4f} | {r["validation_negative_count"]}/{r["test_negative_count"]} |')
        lines+=['','Relative paired improvement, meeting the 0.80 thresholds, and absence of negative curve R2 are separate claims. See paired_source_bootstrap.csv for B-A and C-B source-block intervals.','']
    lines+=['## Definitions and limits','',
        '- A: archived architecture refitted on the nested 131-record subset; B: same architecture on all 158 training records.',
        '- C: training-internal selected classical or constrained residual model. No candidate may bypass the F/T/I guard.',
        '- A/B keep five independent response regressors. C uses four independent outputs and an explicitly reconciled derived quantity.',
        '- The curve metric uses exact piecewise-linear integration over the union of active supports. Nonzero 5% endpoints use one-sided support limits.',
        '- Negative R2 values remain unchanged. Nonnegative force does not imply nonnegative R2.',
        '- Control hyperparameters were inherited as historical architecture choices, but no fitted transformations, coefficients, or ensemble weights are reused.',
        '- Original 189 ordinates are retained. New native Origin histories differ from older digitized present-study reference curves; this is not an algorithmic gain.',
        '- Missing metadata, heterogeneous material-strength definitions, repeated damage and inherited literature evidence limits remain.',
        '- Small-source and degenerate bootstrap samples are not used to manufacture finite precision.',
        '- No manuscript, figure or SHAP result is replaced by this computational package.','']
    (ROOT/'reports/RESULTS_AND_LIMITATIONS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':report()
