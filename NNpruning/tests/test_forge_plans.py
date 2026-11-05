import unittest
from pathlib import Path

from hydra import compose, initialize

from pruning_experiments.plans import build_plan


class ForgePlansTest(unittest.TestCase):
    def plan(self, *overrides):
        with initialize(version_base=None, config_path="../../conf"):
            cfg = compose(config_name="config", overrides=list(overrides))
        return build_plan(cfg, Path("/tmp/forge-plan"))

    def test_benchmark_specific_settings_reach_the_implementation(self):
        plan = self.plan("benchmark=modadd_berman", "branch=berman",
                         "benchmark.fisher_n=500", "runtime.device=cpu")
        self.assertEqual(plan.module, "modadd")
        self.assertIn("--fisher-n", plan.arguments)
        self.assertIn("--device", plan.arguments)
        self.assertEqual(plan.artifact, "berman_iterative_n500.json")

    def test_opt_fisher_settings_reach_the_implementation(self):
        plan = self.plan("benchmark=opt", "branch=fisher_complement",
                         "branch.options.b_sequences=128")
        self.assertEqual(plan.module, "opt_fisher")
        self.assertIn("--fisher-compensation", plan.arguments)
        self.assertIn("complement", plan.arguments)
        self.assertIn("--b-sequences", plan.arguments)

    def test_published_method_requires_its_own_model_and_cuda(self):
        with self.assertRaises(ValueError):
            self.plan("benchmark=published_wanda", "branch=wanda_published")
        plan = self.plan("benchmark=published_wanda", "branch=wanda_published",
                         "runtime.device=cuda:1")
        self.assertEqual(plan.module, "native")
        self.assertIn("decapoda-research/llama-7b-hf", plan.arguments)
        self.assertIn("1", plan.arguments)
        with self.assertRaises(ValueError):
            self.plan("benchmark=published_wanda", "branch=sparsegpt_published",
                      "runtime.device=cuda:0")


if __name__ == "__main__":
    unittest.main()
