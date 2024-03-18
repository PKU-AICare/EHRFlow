from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from Utils.PrintUtils import *
from Utils.PromptTemplateBuilder import PromptTemplateBuilder

from langchain_experimental.plan_and_execute.executors.base import BaseExecutor
from langchain_experimental.plan_and_execute.schema import BaseStepContainer
from Utils.ListStepContainerNew import ListStepContainer
from langchain_experimental.pydantic_v1 import Field
from langchain_experimental.plan_and_execute.schema import Plan


class GraceGPT(Chain):
    planner: Any
    """The planner to use."""
    executor: Any
    """The executor to use."""
    step_container: BaseStepContainer = Field(default_factory=ListStepContainer)
    """The step container to use."""
    input_key: str = "input"
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        plan = self.planner.plan(
            inputs,
            callbacks=run_manager.get_child() if run_manager else None
        )

        for step_index, step in enumerate(plan.steps):
            if step_index >= len(plan.steps):
                break
            step = plan.steps[step_index]
            plan_str = "\n目前的计划设定为：\n" + ' -> '.join(
                [str(num + 1) + '.' + p.value for num, p in enumerate(plan.steps)])
            color_print(plan_str, color=PLAN_COLOR)
            _new_inputs = {
                "previous_steps": self.step_container,
                "current_step": step,
                "objective": inputs[self.input_key]
            }
            new_inputs = {**_new_inputs, **inputs}
            step_str = '现在将会转向这个步骤的求解：' + step.value
            color_print(step_str, color=ROUND_COLOR)
            response = self.executor.run(
                new_inputs
            )
            color_print(response, color=CODE_COLOR)
            self.step_container.add_step(step, response)
            re_plan_inputs = {"previous_steps": self.step_container.get_steps(), "plan": plan, **inputs}
            #  结合执行结果，更新 plan
            new_plan = self.planner.refine_plan(re_plan_inputs)
            plan = Plan(steps=plan.steps[:step_index + 1] + new_plan.steps)
            # print(plan)
        return {self.output_key: self.step_container.get_final_response()}

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        plan = await self.planner.aplan(
            inputs,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        if run_manager:
            await run_manager.on_text(str(plan), verbose=self.verbose)
        for step in plan.steps:
            _new_inputs = {
                "previous_steps": self.step_container,
                "current_step": step,
                "objective": inputs[self.input_key],
            }
            new_inputs = {**_new_inputs, **inputs}
            response = await self.executor.astep(
                new_inputs,
                callbacks=run_manager.get_child() if run_manager else None,
            )
            if run_manager:
                await run_manager.on_text(
                    f"*****\n\nStep: {step.value}", verbose=self.verbose
                )
                await run_manager.on_text(
                    f"\n\nResponse: {response.response}", verbose=self.verbose
                )
            self.step_container.add_step(step, response)
            self.planner.plan()

        return {self.output_key: self.step_container.get_final_response()}
