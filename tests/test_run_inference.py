import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from run_inference import Generator


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return [0] * (3 * len(messages) + int(add_generation_prompt)) + [
            token_id
            for message in messages
            for token_id in self.encode(message["content"])
        ]


class ConversationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.generator = Generator.__new__(Generator)
        self.generator.tokenizer = CharacterTokenizer()
        self.generator.max_model_len = 36
        self.generator.chat_template_kwargs = None
        self.generator.system_prompt = "sys"
        self.generator.system_role_supported = True
        self.generator.n_truncated = 0

    def test_non_positive_fit_budget_returns_empty_text(self):
        self.assertEqual(self.generator._fit("text", 0), "")
        self.assertEqual(self.generator._fit("text", -1), "")
        self.assertLessEqual(len(self.generator._fit("text", 1)), 1)

    def test_complete_multi_turn_template_fits_with_generation_room(self):
        messages = self.generator.build_conversation(
            ["first " * 5, "last " * 5],
            ["answer " * 5],
            max_tokens=12,
        )

        self.assertLessEqual(self.generator._conversation_token_count(messages), 24)
        self.assertEqual(messages[0], {"role": "system", "content": "sys"})
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(self.generator.n_truncated, 1)

    def test_generation_limit_plus_rendered_prompt_fits_context(self):
        vllm = ModuleType("vllm")
        vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        self.generator.max_tokens = 30

        def chat(conversations, params, **kwargs):
            self.generator.conversations = conversations
            self.generator.sampling_params = params
            return []

        self.generator.llm = SimpleNamespace(chat=chat)

        with patch.dict("sys.modules", {"vllm": vllm}):
            self.generator.generate("test", [["prompt " * 20]])

        prompt_tokens = max(
            self.generator._conversation_token_count(conversation)
            for conversation in self.generator.conversations
        )
        self.assertLessEqual(prompt_tokens + self.generator.sampling_params.max_tokens, 36)


if __name__ == "__main__":
    unittest.main()
