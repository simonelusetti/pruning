import unittest
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

from pruning.experiments.plans import build_plan


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
        plan = self.plan("benchmark=opt", "branch=fisher_complement_progressive",
                         "branch.options.b_sequences=128")
        self.assertEqual(plan.module, "opt_fisher")
        self.assertIn("--fisher-compensation", plan.arguments)
        self.assertIn("complement", plan.arguments)
        self.assertIn("--b-sequences", plan.arguments)

        self.assertIn("--fisher-mask-rule", plan.arguments)
        self.assertIn("sparsegpt", plan.arguments)
        self.assertIn("--progressive", plan.arguments)

    def test_opt_progressive_and_sparsegpt_presets_share_calibration(self):
        with initialize(version_base=None, config_path="../../conf"):
            sequential = compose(config_name="config", overrides=[
                "benchmark=opt", "branch=fisher_sparsegpt_refresh"])
            complement = compose(config_name="config", overrides=[
                "benchmark=opt", "branch=fisher_complement_progressive"])
        self.assertEqual(OmegaConf.to_container(sequential.benchmark),
                         OmegaConf.to_container(complement.benchmark))
        sequential_options = OmegaConf.to_container(sequential.branch.options)
        complement_options = OmegaConf.to_container(complement.branch.options)
        for key in ("targets", "progressive", "compensation", "mask_rule",
                    "mu", "cg_max_iterations"):
            sequential_options.pop(key, None)
            complement_options.pop(key, None)
        self.assertEqual(sequential_options, complement_options)

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
