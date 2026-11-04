import unittest

import numpy as np

from pairpulse import (PulseTrain, SauterPulse, sauter_exact_occupation,
                       solve_dirac_mode, solve_qke_mode)


class PairPulseTests(unittest.TestCase):
    def test_sauter_mode_methods_match_exact_solution(self):
        pulse = SauterPulse(0.3, 2.0)
        exact = sauter_exact_occupation(0.0, pulse)
        direct = solve_dirac_mode(0.0, pulse).occupation
        kinetic = solve_qke_mode(0.0, pulse).occupation
        self.assertLess(abs(direct - exact), 2e-9)
        self.assertLess(abs(kinetic - exact), 2e-9)

    def test_qke_invariant_and_fermion_probability_range(self):
        result = solve_qke_mode(0.2, SauterPulse(0.5, 1.5))
        self.assertLess(result.diagnostic, 1e-8)
        self.assertGreaterEqual(result.occupation, 0.0)
        self.assertLessEqual(result.occupation, 1.0)

    def test_zero_field_has_zero_occupation(self):
        pulse = SauterPulse(0.0, 1.0)
        self.assertLess(solve_dirac_mode(0.4, pulse).occupation, 1e-12)
        self.assertEqual(solve_qke_mode(0.4, pulse).occupation, 0.0)
        self.assertEqual(sauter_exact_occupation(0.4, pulse), 0.0)

    def test_pulse_train_is_superposed(self):
        first, second = SauterPulse(0.2, 1.0, -1.0), SauterPulse(0.1, 0.5, 1.0)
        train = PulseTrain([first, second])
        self.assertTrue(np.isclose(train.field(0.0), first.field(0.0) + second.field(0.0)))
        self.assertTrue(np.isclose(train.potential(0.0), first.potential(0.0) + second.potential(0.0)))

    def test_refined_step_cap_retains_exact_benchmark(self):
        pulse = SauterPulse(.3, 2)
        exact = sauter_exact_occupation(.75, pulse)
        for solver in (solve_dirac_mode, solve_qke_mode):
            result = solver(.75, pulse, tail_factor=20, rtol=2e-12,
                            atol=2e-14, max_step_coefficient=.09)
            self.assertLess(abs(result.occupation-exact), 1e-10)

    def test_time_translation_leaves_occupation_unchanged(self):
        for solver in (solve_dirac_mode, solve_qke_mode):
            first = solver(.4, SauterPulse(.3, 1.1)).occupation
            shifted = solver(.4, SauterPulse(.3, 1.1, 7)).occupation
            self.assertLess(abs(first-shifted), 1e-10)

    def test_nonfinite_pulse_parameters_are_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            for kwargs in ({'amplitude':bad,'duration':1},
                           {'amplitude':.3,'duration':bad},
                           {'amplitude':.3,'duration':1,'center':bad}):
                with self.assertRaises(ValueError): SauterPulse(**kwargs)

    def test_invalid_solver_controls_are_rejected(self):
        for solver in (solve_dirac_mode, solve_qke_mode):
            for key in ('mass','charge','tail_factor','rtol','atol','max_step_coefficient'):
                for bad in (0, -1, np.nan, np.inf):
                    with self.assertRaises(ValueError):
                        solver(0, SauterPulse(.3,1), **{key:bad})


if __name__ == "__main__":
    unittest.main()

