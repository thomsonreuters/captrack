import unittest

import numpy as np
import pandas as pd

from compute_metrics import decode_gold, transform_dataframe


class DecodeGoldTests(unittest.TestCase):
    def test_unwraps_singleton_scalar_gold(self):
        self.assertEqual(decode_gold(["A"]), "A")
        self.assertEqual(decode_gold(np.array(["reference summary"])), "reference summary")

    def test_preserves_ifeval_instruction_ids(self):
        transformed = transform_dataframe(
            pd.DataFrame({"outputs": ["response"], "gold": [["instruction_id"]]}),
            task_name="ifeval.subset",
        )

        self.assertEqual(transformed.loc[0, "gold"], ["instruction_id"])

    def test_decodes_json_encoded_list_gold(self):
        self.assertEqual(decode_gold(['["instruction_id"]']), ["instruction_id"])


if __name__ == "__main__":
    unittest.main()
