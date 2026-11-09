"""
Full Benchmark Evaluation Runner & Clustered Bootstrap Engine (Stage 4).
Evaluates the 120-family controlled scientific QA benchmark (480 contexts).
Zero paid API calls. Strictly reproducible offline evaluation.

Protocol:
- Frozen models fit strictly on Train (60 families = 240 contexts).
- Operating thresholds tuned strictly on Validation (24 families = 96 contexts).
- Evaluated on Held-Out Test (36 families = 144 contexts across 36 distinct paper components).
- 2,000 paired cluster bootstrap resamples across connected paper components.
- Disaggregated evaluations:
  1. By condition (supported_candidate, relevant_replacement, duplicate_replacement, natural_retrieval)
  2. By question type (single-passage vs multi-passage across sections)
  3. Controlled mixture vs natural retrieval
  4. Diagnostic context length baseline (passage count / token count)
- Generates risk-coverage curve plots.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Sequence, Tuple, Optional, Union
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, ".")

import plotly.graph_objects as go

from backend.selective.context import FinalContextRecord, Passage
from backend.selective.dataset import (
    QuestionFamily,
    load_pilot_dataset_from_json,
    partition_families_by_manifest
)
from backend.selective.models import (
    AlwaysAcceptBaseline,
    ValidationTunedMaxThreshold,
    FrozenHistoricalGateBaseline,
    LogisticGatingModel,
    LengthDiagnosticBaseline
)
from backend.selective.metrics import (
    compute_evidence_selection_risk,
    compute_type_errors,
    compute_risk_coverage_curve,
    evaluate_at_matched_coverage,
    cluster_bootstrap_evaluation
)
from evaluation.run_selective_eval import (
    flatten_family_contexts,
    _to_json_serializable
)

def plot_risk_coverage_curves(
    curves_dict: Dict[str, Dict[str, Any]],
    output_path_svg: str,
    output_path_html: Optional[str] = None,
    title: str = "Selective Answering: Risk vs. Coverage (Held-Out Test Set)"
):
    """Generate risk-coverage trade-off curves using Plotly and vector SVG."""
    fig = go.Figure()
    
    styles = {
        "B0_AlwaysAccept": ("#95a5a6", "dot", "B0 Always Accept (Full-Coverage Floor)"),
        "B2_FrozenHistoricalGate": ("#e67e22", "none", "B2 Frozen Historical Gate (Operating Point)"),
        "B_Diag_Length": ("#7f8c8d", "dash", "B_Diag Length (Shortcut Diagnostic)"),
        "B1_ValidationTunedThreshold": ("#34495e", "dash", "B1 Max Relevance Score"),
        "M1_Relevance_Raw": ("#e74c3c", "solid", "M1 Relevance (Raw)"),
        "M2_Relevance_Dedup": ("#e67e22", "solid", "M2 Relevance (Dedup)"),
        "M3_Full_Raw": ("#2980b9", "solid", "M3 Full Structure (Raw)"),
        "M4_Full_Dedup": ("#27ae60", "solid", "M4 Full Structure (Dedup)")
    }
    
    for mname, curve in curves_dict.items():
        if mname not in styles:
            continue
        color, lstyle, label = styles[mname]
        
        if mname == "B0_AlwaysAccept":
            risk_val = curve.get("risk", 0.0)
            fig.add_trace(go.Scatter(
                x=[0.0, 1.0], y=[risk_val, risk_val],
                mode="lines",
                line=dict(color=color, dash="dot", width=2),
                name=f"{label} (Risk={risk_val:.3f})"
            ))
        elif mname == "B2_FrozenHistoricalGate":
            cov_val = curve.get("coverage", 0.0)
            risk_val = curve.get("risk", 0.0)
            fig.add_trace(go.Scatter(
                x=[cov_val], y=[risk_val],
                mode="markers",
                marker=dict(color=color, symbol="square", size=12),
                name=f"{label} (Cov={cov_val:.2f}, Risk={risk_val:.3f})"
            ))
        else:
            covs = [pt["coverage"] for pt in curve["points"]]
            risks = [pt.get("expected_risk", pt.get("risk", 0.0)) for pt in curve["points"]]
            aurc_val = curve.get("aurc", 0.0)
            fig.add_trace(go.Scatter(
                x=covs, y=risks,
                mode="lines",
                line=dict(color=color, dash=lstyle, width=2.5),
                name=f"{label} (AURC={aurc_val:.4f})"
            ))
            
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Arial")),
        xaxis=dict(title="Coverage (Fraction of Questions Answered)", range=[0.0, 1.02], dtick=0.2),
        yaxis=dict(title="Evidence Selection Risk", range=[-0.02, 1.02], dtick=0.2),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1),
        template="plotly_white",
        width=850,
        height=550
    )
    
    if output_path_html:
        os.makedirs(os.path.dirname(output_path_html) or ".", exist_ok=True)
        fig.write_html(output_path_html)
        print(f"Saved interactive plot to {output_path_html}")
        
    os.makedirs(os.path.dirname(output_path_svg) or ".", exist_ok=True)
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="580" viewBox="0 0 900 580" style="background:#ffffff; font-family:sans-serif;">',
        f'<text x="450" y="32" font-size="16" font-weight="bold" text-anchor="middle" fill="#2c3e50">{title}</text>',
        f'<g transform="translate(80, 50)">',
        f'  <line x1="0" y1="450" x2="520" y2="450" stroke="#333" stroke-width="1.5" />',
        f'  <line x1="0" y1="0" x2="0" y2="450" stroke="#333" stroke-width="1.5" />'
    ]
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x_pos = tick * 520
        y_pos = 450 - (tick * 450)
        svg_lines.append(f'  <line x1="{x_pos}" y1="0" x2="{x_pos}" y2="450" stroke="#eee" stroke-width="1" />')
        svg_lines.append(f'  <line x1="0" y1="{y_pos}" x2="520" y2="{y_pos}" stroke="#eee" stroke-width="1" />')
        svg_lines.append(f'  <text x="{x_pos}" y="470" font-size="11" text-anchor="middle" fill="#666">{tick:.1f}</text>')
        svg_lines.append(f'  <text x="-12" y="{y_pos+4}" font-size="11" text-anchor="end" fill="#666">{tick:.1f}</text>')
        
    svg_lines.append(f'  <text x="260" y="505" font-size="12" font-weight="bold" text-anchor="middle" fill="#2c3e50">Coverage (Fraction of Questions Answered)</text>')
    svg_lines.append(f'  <text x="-40" y="225" font-size="12" font-weight="bold" text-anchor="middle" fill="#2c3e50" transform="rotate(-90 -40 225)">Evidence Selection Risk</text>')

    for mname, curve in curves_dict.items():
        if mname not in styles:
            continue
        color, lstyle, label = styles[mname]
        if mname == "B0_AlwaysAccept":
            r_y = 450 - (curve.get("risk", 0.0) * 450)
            svg_lines.append(f'  <line x1="0" y1="{r_y}" x2="520" y2="{r_y}" stroke="{color}" stroke-dasharray="4,4" stroke-width="2" />')
        elif mname == "B2_FrozenHistoricalGate":
            c_x = curve.get("coverage", 0.0) * 520
            r_y = 450 - (curve.get("risk", 0.0) * 450)
            svg_lines.append(f'  <rect x="{c_x-5}" y="{r_y-5}" width="10" height="10" fill="{color}" stroke="#fff" stroke-width="1.5" />')
        else:
            pts = curve.get("points", [])
            if pts:
                d_str = " ".join([f'{"M" if i==0 else "L"} {p["coverage"]*520:.1f} {450-(p.get("expected_risk", p.get("risk", 0.0)))*450:.1f}' for i, p in enumerate(pts)])
                dash_attr = ' stroke-dasharray="6,4"' if lstyle == "dash" else ""
                svg_lines.append(f'  <path d="{d_str}" fill="none" stroke="{color}" stroke-width="2.5"{dash_attr} />')

    svg_lines.append(f'  <g transform="translate(540, 20)">')
    svg_lines.append(f'    <rect x="0" y="0" width="260" height="230" fill="#fdfdfd" stroke="#ddd" rx="4" />')
    l_idx = 0
    for mname in ["M4_Full_Dedup", "M3_Full_Raw", "M2_Relevance_Dedup", "M1_Relevance_Raw", "B1_ValidationTunedThreshold", "B_Diag_Length", "B2_FrozenHistoricalGate", "B0_AlwaysAccept"]:
        if mname not in curves_dict or mname not in styles:
            continue
        color, lstyle, label = styles[mname]
        y_l = 25 + l_idx * 25
        aurc_str = f" (AURC={curves_dict[mname].get('aurc', 0.0):.4f})" if "aurc" in curves_dict[mname] else ""
        if mname == "B2_FrozenHistoricalGate":
            svg_lines.append(f'    <rect x="10" y="{y_l-4}" width="8" height="8" fill="{color}" />')
            svg_lines.append(f'    <text x="26" y="{y_l+3}" font-size="10" fill="#333">B2 Historical Gate (Cov={curves_dict[mname].get("coverage",0):.2f})</text>')
        elif mname == "B0_AlwaysAccept":
            svg_lines.append(f'    <line x1="8" y1="{y_l}" x2="22" y2="{y_l}" stroke="{color}" stroke-dasharray="3,3" stroke-width="2" />')
            svg_lines.append(f'    <text x="26" y="{y_l+3}" font-size="10" fill="#333">B0 Always Accept (Risk={curves_dict[mname].get("risk",0):.3f})</text>')
        else:
            dash_attr = ' stroke-dasharray="4,3"' if lstyle == "dash" else ""
            svg_lines.append(f'    <line x1="8" y1="{y_l}" x2="22" y2="{y_l}" stroke="{color}" stroke-width="2"{dash_attr} />')
            svg_lines.append(f'    <text x="26" y="{y_l+3}" font-size="10" fill="#333">{mname}{aurc_str}</text>')
        l_idx += 1
    svg_lines.append(f'  </g>')
    svg_lines.append(f'</g>')
    svg_lines.append(f'</svg>')
    
    with open(output_path_svg, "w", encoding="utf-8") as f_svg:
        f_svg.write("\n".join(svg_lines))
    print(f"Saved SVG risk-coverage plot to {output_path_svg}")


def run_evaluation():
    t_start = time.time()
    print("=== Running Full Controlled Benchmark Evaluation (Stage 4) ===")
    
    dataset_path = "data/metadata/full_benchmark_120families_annotated.json"
    manifest_path = "data/metadata/full_benchmark_split_manifest.json"
    frozen_models_path = "data/metadata/frozen_models_train_val.json"
    
    # 1. Load dataset & manifest
    families = load_pilot_dataset_from_json(dataset_path)
    train_fams, val_fams, test_fams, split_meta = partition_families_by_manifest(families, manifest_path)
    fam_to_cid = split_meta["family_to_component_id"]
    
    print(f"Dataset loaded: 120 families across {len(set(fam_to_cid.values()))} components.")
    print(f"  Train: {len(train_fams)} families | Val: {len(val_fams)} families | Test: {len(test_fams)} families")
    
    # Flatten records
    train_recs, train_lbls, train_rids, train_fids, train_cids = flatten_family_contexts(train_fams, fam_to_cid)
    val_recs, val_lbls, val_rids, val_fids, val_cids = flatten_family_contexts(val_fams, fam_to_cid)
    test_recs, test_lbls, test_rids, test_fids, test_cids = flatten_family_contexts(test_fams, fam_to_cid)
    
    valid_train_lbls = [int(x) for x in train_lbls]
    valid_val_lbls = [int(x) for x in val_lbls]
    valid_test_lbls = [int(x) for x in test_lbls]
    
    print(f"Flattened contexts:")
    print(f"  Train: {len(train_recs)} (Pos={sum(valid_train_lbls)}, Neg={len(valid_train_lbls)-sum(valid_train_lbls)})")
    print(f"  Val:   {len(val_recs)} (Pos={sum(valid_val_lbls)}, Neg={len(valid_val_lbls)-sum(valid_val_lbls)})")
    print(f"  Test:  {len(test_recs)} (Pos={sum(valid_test_lbls)}, Neg={len(valid_test_lbls)-sum(valid_test_lbls)})")
    
    # 2. Fit models on Train & tune thresholds on Val
    b_diag = LengthDiagnosticBaseline()
    b_diag.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    b_diag.fit_validation_thresholds(val_recs)
    
    b1 = ValidationTunedMaxThreshold()
    b1.fit_validation_thresholds(val_recs, valid_val_lbls)
    
    m1 = LogisticGatingModel(name="M1_Relevance_Raw", feature_mode="relevance_only", deduplicate=False)
    m1.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    m1.fit_validation_thresholds(val_recs)
    
    m2 = LogisticGatingModel(name="M2_Relevance_Dedup", feature_mode="relevance_only", deduplicate=True)
    m2.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    m2.fit_validation_thresholds(val_recs)
    
    m3 = LogisticGatingModel(name="M3_Full_Raw", feature_mode="combined", deduplicate=False)
    m3.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    m3.fit_validation_thresholds(val_recs)
    
    m4 = LogisticGatingModel(name="M4_Full_Dedup", feature_mode="combined", deduplicate=True)
    m4.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    m4.fit_validation_thresholds(val_recs)
    
    b0 = AlwaysAcceptBaseline()
    b2 = FrozenHistoricalGateBaseline()
    
    b0_test_decisions = [b0.decide(r) for r in test_recs]
    b0_risk = compute_evidence_selection_risk(b0_test_decisions, valid_test_lbls)
    
    b2_test_decisions = [b2.decide(r) for r in test_recs]
    b2_cov = float(np.mean(b2_test_decisions))
    b2_risk = compute_evidence_selection_risk(b2_test_decisions, valid_test_lbls)
    b2_t1, b2_t2 = compute_type_errors(b2_test_decisions, valid_test_lbls)
    
    ranking_models = {
        "B_Diag_Length": b_diag,
        "B1_ValidationTunedThreshold": b1,
        "M1_Relevance_Raw": m1,
        "M2_Relevance_Dedup": m2,
        "M3_Full_Raw": m3,
        "M4_Full_Dedup": m4
    }
    
    scores_dict = {}
    preds_dict = {
        "B0_AlwaysAccept": b0_test_decisions,
        "B2_FrozenHistoricalGate": b2_test_decisions
    }
    curves_for_plot = {
        "B0_AlwaysAccept": {"risk": b0_risk},
        "B2_FrozenHistoricalGate": {"coverage": b2_cov, "risk": b2_risk}
    }
    
    eval_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": "1.0_frozen",
        "benchmark_scale": {
            "total_families": len(families),
            "train_families": len(train_fams),
            "val_families": len(val_fams),
            "test_families": len(test_fams),
            "train_contexts": len(train_recs),
            "val_contexts": len(val_recs),
            "test_contexts": len(test_recs),
            "test_clusters_count": len(set(test_cids))
        },
        "baselines": {
            "B0_AlwaysAccept": {
                "achieved_coverage": 1.0,
                "evidence_selection_risk": round(b0_risk, 4),
                "type1_error_insufficient_accepted": round(1.0 - np.mean(valid_test_lbls), 4),
                "type2_error_sufficient_rejected": 0.0
            },
            "B2_FrozenHistoricalGate": {
                "achieved_coverage": round(b2_cov, 4),
                "accepted_count": int(np.sum(b2_test_decisions)),
                "evidence_selection_risk": round(b2_risk, 4) if b2_risk is not None else None,
                "type1_error_insufficient_accepted": round(b2_t1, 4) if b2_t1 is not None else None,
                "type2_error_sufficient_rejected": round(b2_t2, 4) if b2_t2 is not None else None
            }
        },
        "models": {}
    }
    
    target_coverages = [0.50, 0.75, 0.90]
    observed_aurcs = {}
    
    print("\n=== Primary Continuous Ranking Model Evaluations (Test Set, N=144) ===")
    for name, m in ranking_models.items():
        test_scores = [float(m.predict_score(r)) for r in test_recs]
        scores_dict[name] = test_scores
        preds_dict[name] = [m.decide(r) for r in test_recs]
        
        curve = compute_risk_coverage_curve(test_scores, valid_test_lbls)
        aurc_val = round(curve["aurc"], 4)
        observed_aurcs[name] = aurc_val
        curves_for_plot[name] = curve
        
        matched_desc = evaluate_at_matched_coverage(curve, target_coverages)
        
        # Deployable operating points using validation-frozen thresholds
        deployable_ops = {}
        for cov in target_coverages:
            cov_key = f"target_cov_{cov:.2f}"
            th = m.operating_thresholds.get(cov, m.operating_thresholds.get(cov_key))
            op_preds = [1 if s >= th else 0 for s in test_scores]
            op_cov = float(np.mean(op_preds))
            op_risk = compute_evidence_selection_risk(op_preds, valid_test_lbls)
            op_t1, op_t2 = compute_type_errors(op_preds, valid_test_lbls)
            
            deployable_ops[cov_key] = {
                "frozen_validation_threshold": round(float(th), 4),
                "achieved_test_coverage": round(op_cov, 4),
                "accepted_count": int(np.sum(op_preds)),
                "evidence_selection_risk": round(op_risk, 4) if op_risk is not None else None,
                "type1_error_insufficient_accepted": round(op_t1, 4) if op_t1 is not None else None,
                "type2_error_sufficient_rejected": round(op_t2, 4) if op_t2 is not None else None
            }
            # Add specific operating point for discrete decisions in bootstrap
            preds_dict[f"{name}@{cov_key}"] = op_preds
            
        eval_results["models"][name] = {
            "aurc": aurc_val,
            "matched_coverage_descriptive": matched_desc,
            "deployable_operating_points": deployable_ops
        }
        print(f"  {name:28s} | AURC={aurc_val:.4f} | Cov@0.50 Risk={deployable_ops['target_cov_0.50']['evidence_selection_risk']} | Cov@0.75 Risk={deployable_ops['target_cov_0.75']['evidence_selection_risk']}")

    # 3. Clustered Bootstrap (2,000 resamples across the 36 test paper components)
    print("\nRunning 2,000 paired cluster bootstrap resamples across 36 test paper components...")
    boot_res = cluster_bootstrap_evaluation(
        cluster_ids=test_cids,
        predictions_dict=preds_dict,
        scores_dict=scores_dict,
        labels=valid_test_lbls,
        n_bootstrap=2000,
        random_state=42,
        observed_aurcs=observed_aurcs
    )
    eval_results["cluster_bootstrap_2000"] = boot_res
    
    # Print Primary Hypothesis Testing Deltas
    print("\n=== Primary Hypothesis Testing (AURC Differences, 2,000 Paired Bootstrap Resamples) ===")
    primary_pairs = [
        ("M4_Full_Dedup_minus_M2_Relevance_Dedup", "Structural Benefit post-dedup (M4 - M2)"),
        ("M4_Full_Dedup_minus_M3_Full_Raw", "Deduplication Benefit with Structure (M4 - M3)"),
        ("M2_Relevance_Dedup_minus_M1_Relevance_Raw", "Deduplication on Pure Relevance (M2 - M1)"),
        ("B1_ValidationTunedThreshold_minus_M4_Full_Dedup", "Max-Threshold Baseline vs M4 (B1 - M4)"),
        ("B_Diag_Length_minus_M4_Full_Dedup", "Length Diagnostic Baseline vs M4 (Diag - M4)")
    ]
    
    for pair_key, desc in primary_pairs:
        pair_data = boot_res["paired_differences"].get(pair_key, {})
        obs_d = pair_data.get("observed_delta")
        boot_d = pair_data.get("bootstrap_mean_delta")
        ci = pair_data.get("aurc_delta_ci_95")
        print(f"  {desc:48s} | Observed Delta={obs_d:+.4f} | Bootstrap Mean={boot_d:+.4f} | 95% CI={ci}")

    # 4. Disaggregated Evaluations
    print("\n=== Disaggregated Subgroup Evaluations ===")
    subgroup_results = {}
    
    # A. By Condition
    conditions = ["supported_candidate", "relevant_replacement", "duplicate_replacement", "natural_retrieval"]
    subgroup_results["by_condition"] = {}
    for cond in conditions:
        cond_idx = [i for i, rid in enumerate(test_rids) if rid.endswith(f":{cond}")]
        c_lbls = [valid_test_lbls[i] for i in cond_idx]
        c_res = {}
        for mname, m in ranking_models.items():
            c_scores = [scores_dict[mname][i] for i in cond_idx]
            c_curve = compute_risk_coverage_curve(c_scores, c_lbls)
            c_res[mname] = {"aurc": round(c_curve["aurc"], 4), "mean_score": round(float(np.mean(c_scores)), 4)}
        subgroup_results["by_condition"][cond] = {
            "n_contexts": len(cond_idx),
            "pos_count": sum(c_lbls),
            "models": c_res
        }
        print(f"  Condition '{cond}' (N={len(cond_idx)}, Pos={sum(c_lbls)}): M4 AURC={c_res['M4_Full_Dedup']['aurc']:.4f}, M2 AURC={c_res['M2_Relevance_Dedup']['aurc']:.4f}")
        
    # B. By Question Type (Single vs Multi)
    test_specs = {s["family_id"]: s["is_multi_passage"] for s in json.load(open("data/metadata/benchmark_120_question_specs.json"))["test"]}
    single_idx = [i for i, fid in enumerate(test_fids) if not test_specs[fid]]
    multi_idx = [i for i, fid in enumerate(test_fids) if test_specs[fid]]
    
    subgroup_results["by_question_type"] = {
        "single_passage": {
            "n_contexts": len(single_idx),
            "pos_count": sum(valid_test_lbls[i] for i in single_idx),
            "models": {mname: round(compute_risk_coverage_curve([scores_dict[mname][i] for i in single_idx], [valid_test_lbls[i] for i in single_idx])["aurc"], 4) for mname in ranking_models}
        },
        "multi_passage": {
            "n_contexts": len(multi_idx),
            "pos_count": sum(valid_test_lbls[i] for i in multi_idx),
            "models": {mname: round(compute_risk_coverage_curve([scores_dict[mname][i] for i in multi_idx], [valid_test_lbls[i] for i in multi_idx])["aurc"], 4) for mname in ranking_models}
        }
    }
    print(f"  Single-Passage (N={len(single_idx)}): M4 AURC={subgroup_results['by_question_type']['single_passage']['models']['M4_Full_Dedup']:.4f}")
    print(f"  Multi-Passage  (N={len(multi_idx)}): M4 AURC={subgroup_results['by_question_type']['multi_passage']['models']['M4_Full_Dedup']:.4f}")
    
    # C. Controlled Mixture vs Natural Retrieval
    ctrl_idx = [i for i, rid in enumerate(test_rids) if not rid.endswith(":natural_retrieval")]
    nat_idx = [i for i, rid in enumerate(test_rids) if rid.endswith(":natural_retrieval")]
    
    subgroup_results["controlled_vs_natural"] = {
        "controlled_3condition_mixture": {
            "n_contexts": len(ctrl_idx),
            "pos_count": sum(valid_test_lbls[i] for i in ctrl_idx),
            "models": {mname: round(compute_risk_coverage_curve([scores_dict[mname][i] for i in ctrl_idx], [valid_test_lbls[i] for i in ctrl_idx])["aurc"], 4) for mname in ranking_models}
        },
        "natural_retrieval": {
            "n_contexts": len(nat_idx),
            "pos_count": sum(valid_test_lbls[i] for i in nat_idx),
            "models": {mname: round(compute_risk_coverage_curve([scores_dict[mname][i] for i in nat_idx], [valid_test_lbls[i] for i in nat_idx])["aurc"], 4) for mname in ranking_models}
        }
    }
    eval_results["disaggregated_subgroups"] = subgroup_results
    
    # 5. Generate plots
    plot_path_svg = "data/metadata/risk_coverage_curve_full_benchmark.svg"
    plot_path_html = "data/metadata/risk_coverage_curve_full_benchmark.html"
    plot_risk_coverage_curves(curves_for_plot, plot_path_svg, plot_path_html)
    
    # 6. Save final results JSON
    eval_results["total_elapsed_sec"] = round(time.time() - t_start, 3)
    out_results_path = "data/metadata/full_benchmark_eval_results.json"
    with open(out_results_path, "w", encoding="utf-8") as fp:
        json.dump(_to_json_serializable(eval_results), fp, indent=2)
    print(f"\nFinal full benchmark evaluation results saved to {out_results_path}.")
    print(f"Total evaluation elapsed: {eval_results['total_elapsed_sec']}s.")

if __name__ == "__main__":
    run_evaluation()
