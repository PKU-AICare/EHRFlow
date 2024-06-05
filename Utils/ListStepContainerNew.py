from typing import List, Tuple

from langchain_experimental.plan_and_execute.schema import BaseStepContainer, Step, StepResponse
from langchain_experimental.pydantic_v1 import BaseModel, Field


class ListStepContainer(BaseStepContainer):
    """Container for List of steps."""

    steps: List[Tuple[Step, StepResponse]] = Field(default_factory=list)
    """The steps."""

    def add_step(self, step: Step, step_response: StepResponse) -> None:
        self.steps.append((step, step_response))

    def get_steps(self) -> List[Tuple[Step, StepResponse]]:
        return self.steps

    def get_final_response(self):
        return self.steps[-1][1]
