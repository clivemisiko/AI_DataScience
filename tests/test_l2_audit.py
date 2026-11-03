"""Contract tests for the Track L2 fairness and explainability deliverables."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_l2_notebook_contains_two_prediction_explanations():
    notebook = json.loads((ROOT / "l2_fairness_explainability.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for line in cell["source"]
    )
    assert "explain_indices" in source
    assert "shap.Explainer" in source
    assert "shap.plots.waterfall" in source


def test_l2_report_has_measured_findings_and_actionable_mitigation():
    report = (ROOT / "l2_fairness_audit_report.md").read_text(encoding="utf-8")
    assert "0.0227" in report
    assert "minimum of 30 positive cases" in report
    assert "human review" in report
    assert "Block deployment" in report
