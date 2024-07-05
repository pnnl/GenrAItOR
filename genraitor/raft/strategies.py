from enum import Enum


class TrainingStrategy(str, Enum):
    ORPO = "orpo"
    SFT = "sft"

    @staticmethod
    def from_str(value):
        match value:
            case "orpo":
                return TrainingStrategy.ORPO
            case _:
                return TrainingStrategy.SFT

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
