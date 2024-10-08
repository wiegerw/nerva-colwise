# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List, Optional

import nervalibcolwise

from nervacolwise.utilities import parse_function_call


class LearningRateScheduler(nervalibcolwise.learning_rate_scheduler):
    pass


class TimeBasedScheduler(nervalibcolwise.time_based_scheduler):
    def __init__(self, lr: float, decay: float):
        super().__init__(lr, decay)

    def __str__(self):
        return f'TimeBasedScheduler(lr={self.lr}, decay={self.decay})'


class StepBasedScheduler(nervalibcolwise.step_based_scheduler):
    def __init__(self, lr: float, drop_rate: float, change_rate: float):
        super().__init__(lr, drop_rate, change_rate)

    def __str__(self):
        return f'StepBasedScheduler(lr={self.lr}, drop_rate={self.drop_rate}, change_rate={self.change_rate})'


class MultiStepLRScheduler(nervalibcolwise.multi_step_lr_scheduler):
    def __init__(self, lr: float, milestones: List[int], gamma: float):
        super().__init__(lr, milestones, gamma)

    def __str__(self):
        return f'MultiStepLRScheduler(lr={self.lr}, milestones={self.milestones}, gamma={self.gamma})'


class ExponentialScheduler(nervalibcolwise.exponential_scheduler):
    def __init__(self, lr: float, change_rate: float):
        super().__init__(lr, change_rate)

    def __str__(self):
        return f'ExponentialScheduler(lr={self.lr}, change_rate={self.change_rate})'


def parse_learning_rate_scheduler(text: str) -> Optional[LearningRateScheduler]:
    if not text:
        return None

    func = parse_function_call(text)
    if func.name == 'TimeBased':
        lr = func.as_float('lr')
        decay = func.as_float('decay')
        return TimeBasedScheduler(lr, decay)
    elif func.name == 'StepBased':
        lr = func.as_float('lr')
        drop_rate = func.as_float('drop_rate')
        change_rate = func.as_float('change_rate')
        return StepBasedScheduler(lr, drop_rate, change_rate)
    elif func.name == 'MultiStepLR':
        lr = func.as_float('lr')
        milestones = [int(x) for x in func.as_string('milestones').split('|')]
        gamma = func.as_float('gamma')
        return MultiStepLRScheduler(lr, milestones, gamma)
    elif func.name == 'Exponential':
        lr = func.as_float('lr')
        change_rate = func.as_float('change_rate')
        return ExponentialScheduler(lr, change_rate)
    raise RuntimeError(f"Could not parse learning rate scheduler '{text}'")
