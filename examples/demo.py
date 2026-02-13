#!/usr/bin/env python3
"""
Run one or more ranking_gam demos.

Individual demos (each is self-contained):
    python examples/demo_gam.py           # GAM / GA2M / ContextGAM + distillation
    python examples/demo_submodular.py    # SubmodularRankingGAM (diversity)
    python examples/demo_multi.py         # Multi-Objective (weight scenarios)
    python examples/demo_gbdt.py          # GBDT residual boosting (3-stage)
    python examples/demo_boost.py         # Item boosting via response curves

This wrapper runs all or selected demos with shared args:
    python examples/demo.py                           # run all
    python examples/demo.py --demo gam                # just GAM
    python examples/demo.py --demo gam,gbdt           # GAM + GBDT
    python examples/demo.py --demo gam --context      # ContextGAM
    python examples/demo.py --demo gam --loss ndcg2pp --activation silu
"""

import argparse
import sys

from _common import (
    add_common_args, load_data, print_config, resolve_args,
)

DEMO_CHOICES = {"gam", "submodular", "multi", "gbdt", "boost"}


def main():
    parser = argparse.ArgumentParser(
        description="ranking_gam demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo choices:
  gam          GAM / GA2M / ContextGAM with distillation
  submodular   SubmodularRankingGAM with diversity evaluation
  multi        Multi-Objective Ranking GAM with weight scenarios
  gbdt         GBDT residual boosting (3-stage pipeline)
  boost        Item boosting via response curve manipulation (cold-start)

Examples:
  python examples/demo.py --demo gam --loss ndcg2pp --activation silu
  python examples/demo.py --demo gam --context --lr-schedule plateau
  python examples/demo.py --demo gbdt --loss ndcg2pp
  python examples/demo.py --demo gam,submodular --epochs 30
        """,
    )
    parser.add_argument(
        "--demo", type=str, default="all",
        help="comma-separated demos: gam,submodular,multi,gbdt,all (default: all)",
    )
    add_common_args(parser)
    parser.add_argument("--queries-per-epoch", type=int, default=400)
    # GAM-specific
    parser.add_argument("--context", action="store_true",
                        help="use ContextGAM (GAM demo only)")
    parser.add_argument("--ga2m", action="store_true",
                        help="use GA2M with interactions (GAM demo only)")
    parser.add_argument("--ga2m-pairs", type=int, default=20)
    parser.add_argument("--tower-dropout", type=float, default=0.0)
    parser.add_argument("--output-norm", action="store_true")
    # Boost-specific
    parser.add_argument("--boost-feature", type=int, default=0,
                        help="feature index to boost (boost demo only)")
    parser.add_argument("--boost-percentile", type=float, default=75,
                        help="target percentile for boosted items (boost demo only)")
    parser.add_argument("--cold-start-fraction", type=float, default=0.1,
                        help="fraction of items to treat as cold-start (boost demo only)")
    args = resolve_args(parser.parse_args())

    # Parse demo selection
    if args.demo == "all":
        demos = DEMO_CHOICES
    else:
        demos = {d.strip() for d in args.demo.split(",")}
        unknown = demos - DEMO_CHOICES
        if unknown:
            parser.error(f"Unknown demo(s): {unknown}. Choose from: {DEMO_CHOICES}")

    extra = []
    if args.context:
        extra.append("context_weights")
    if args.ga2m:
        extra.append(f"ga2m(pairs={args.ga2m_pairs})")
    print(f"Running demos: {', '.join(sorted(demos))}")
    print_config(args, extra)

    data = load_data(dataset=args.dataset, data_dir=args.data_dir)
    results = {}

    if "gam" in demos:
        from demo_gam import run as run_gam
        results.update(run_gam(data, args))

    if "submodular" in demos:
        from demo_submodular import run as run_submodular
        results.update(run_submodular(data, args))

    if "multi" in demos:
        from demo_multi import run as run_multi
        run_multi(data, args)

    if "gbdt" in demos:
        from demo_gbdt import run as run_gbdt
        results.update(run_gbdt(data, args))

    if "boost" in demos:
        from demo_boost import main as run_boost_main
        # boost demo has its own argparse, so we just call it with sys.argv
        # For the wrapper, we re-run with the shared args
        import sys
        saved_argv = sys.argv
        boost_argv = ["demo_boost.py",
                       f"--dataset={args.dataset}", f"--data-dir={args.data_dir}",
                       f"--epochs={args.epochs}", f"--k={args.k}",
                       f"--loss={args.loss}", f"--activation={args.activation}",
                       f"--lr-schedule={args.lr_schedule}",
                       f"--boost-feature={args.boost_feature}",
                       f"--boost-percentile={args.boost_percentile}",
                       f"--cold-start-fraction={args.cold_start_fraction}"]
        if not args.transforms:
            boost_argv.append("--no-transforms")
        if not args.residual:
            boost_argv.append("--no-residual")
        sys.argv = boost_argv
        run_boost_main()
        sys.argv = saved_argv

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        for name, ndcg in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name:<28} NDCG@{args.k} = {ndcg:.4f}")


if __name__ == "__main__":
    main()
