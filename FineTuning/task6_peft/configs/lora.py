from custom_helpers import get_config
from .base import BaseConfig
from peft import (
    LoraConfig,
    TaskType,
    PeftConfig
)
from typing import Dict


class PeftLoRAConfig(BaseConfig):
    """Configuration class for LoRA PEFT model training."""
    
    @property
    def model_config(self) -> Dict:
        return {
            "model_name": self.conf["model"],
            "quantize": False,
            "is_prompt_tuning": False
        }
    
    @property
    def peft_config(self) -> PeftConfig:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.conf.lora.r,
            lora_alpha=self.conf.lora.lora_alpha,
            lora_dropout=self.conf.lora.lora_dropout,
            target_modules=self.conf.lora.target_modules
        )
            

    @property
    def training_config(self) -> Dict:
        return {
            **self.DEFAULT_TRAINING_CONFIG,
            "output_dir": self.conf.exp.lora_dir
        }
