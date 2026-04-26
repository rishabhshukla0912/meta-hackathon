# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
import unittest
from unittest import mock

from training import lora_setup


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLoraConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePeftModel:
    def __init__(self, base_model, peft_config):
        self.base_model = base_model
        self.peft_config = {"default": peft_config}
        self.added_adapters = []
        self.active_adapter = None

    def add_adapter(self, adapter_name, peft_config):
        self.peft_config[adapter_name] = peft_config
        self.added_adapters.append(adapter_name)

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name


def fake_get_peft_model(model, peft_config):
    return FakePeftModel(model, peft_config)


class LoRASetupTest(unittest.TestCase):
    def test_base_model_kwargs_without_quantization(self):
        kwargs = lora_setup._base_model_kwargs(
            load_in_4bit=False,
            device_map="auto",
        )

        self.assertEqual(kwargs["torch_dtype"], "auto")
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertNotIn("load_in_4bit", kwargs)
        self.assertNotIn("quantization_config", kwargs)

    def test_base_model_kwargs_uses_quantization_config_for_4bit(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bfloat16"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig

        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            kwargs = lora_setup._base_model_kwargs(
                load_in_4bit=True,
                device_map="auto",
            )

        quantization_config = kwargs["quantization_config"]
        self.assertNotIn("load_in_4bit", kwargs)
        self.assertIsInstance(quantization_config, FakeBitsAndBytesConfig)
        self.assertEqual(
            quantization_config.kwargs,
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
        )

    def test_attach_lora_adapters_keeps_default_for_trl(self):
        fake_peft = types.ModuleType("peft")
        fake_peft.LoraConfig = FakeLoraConfig
        fake_peft.get_peft_model = fake_get_peft_model

        with mock.patch.dict(sys.modules, {"peft": fake_peft}):
            peft_model = lora_setup.attach_lora_adapters(model=object())

        self.assertIn("default", peft_model.peft_config)
        self.assertEqual(peft_model.added_adapters, ["A", "S", "B"])
        self.assertEqual(peft_model.active_adapter, "A")


if __name__ == "__main__":
    unittest.main()
