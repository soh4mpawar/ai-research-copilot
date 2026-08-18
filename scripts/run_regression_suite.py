"""
Master Regression Suite & CI Gate Runner.
Usage:
    python scripts/run_regression_suite.py                # Runs all unit & regression tests
    python scripts/run_regression_suite.py --with-ragas   # Runs unit tests + full 40-sample RAGAS benchmark gate
"""

import unittest
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_unit_regression_suite() -> bool:
    """Run all test modules in tests/ directory."""
    print("=" * 80)
    print("RUNNING AI RESEARCH COPILOT REGRESSION TEST SUITE")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_ragas_ci_gate() -> bool:
    """Run full 40-sample RAGAS evaluation and verify PRD §8 thresholds."""
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL 40-SAMPLE RAGAS EVALUATION CI GATE (PRD §8)")
    print("=" * 80)

    from evaluation.evaluate_ragas import main as run_eval
    try:
        results = run_eval()
        faithfulness = results.get("faithfulness", 0.0)
        context_precision = results.get("context_precision", 0.0)
        context_recall = results.get("context_recall", 0.0)
        answer_relevancy = results.get("answer_relevancy", 0.0)

        targets = {
            "Faithfulness": (faithfulness, 0.70),
            "Context Precision": (context_precision, 0.60),
            "Context Recall": (context_recall, 0.60),
            "Answer Relevancy": (answer_relevancy, 0.70),
        }

        all_passed = True
        print("\n--- PRD §8 BENCHMARK GATE SUMMARY ---")
        for metric, (val, target) in targets.items():
            status = "PASS" if val >= target else "FAIL"
            if val < target:
                all_passed = False
            print(f"  • {metric:20s}: {val:.4f} (Target: >={target:.2f}) -> [{status}]")

        return all_passed
    except Exception as e:
        print(f"[CI Gate Error] RAGAS evaluation failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="AI Research Copilot CI Regression Suite")
    parser.add_argument("--with-ragas", action="store_true", help="Include full 40-sample RAGAS benchmark gate")
    args = parser.parse_args()

    t0 = time.time()
    unit_ok = run_unit_regression_suite()

    ragas_ok = True
    if args.with_ragas:
        ragas_ok = run_ragas_ci_gate()

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    if unit_ok and ragas_ok:
        print(f"ALL REGRESSION TESTS PASSED SUCCESSFULLY in {elapsed:.2f}s! [STATUS: GREEN]")
        print("=" * 80)
        sys.exit(0)
    else:
        print(f"REGRESSION SUITE FAILED in {elapsed:.2f}s! [STATUS: RED]")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
