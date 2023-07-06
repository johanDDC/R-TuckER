import math


class RegularizationCoeffPolicy:
    def __init__(self, base_val, num_steps):
        self.base_val = base_val
        self.num_steps = num_steps
        self.val = base_val

    def step(self):
        pass


class IntervalPolicy(RegularizationCoeffPolicy):
    def __init__(self, base_val, num_steps, final_val):
        super().__init__(base_val, num_steps)
        self.final_val = final_val


class SimpleDecreasingPolicy(IntervalPolicy):
    def __init__(self, base_val, num_steps, final_val, strategy="linear"):
        super().__init__(base_val, num_steps, final_val)
        self.strategy = strategy
        if self.strategy == "linear":
            self.step_size = (base_val - final_val) / num_steps
        elif self.strategy == "exp":
            self.step_size = math.pow(self.final_val / self.base_val, 1 / self.num_steps)
        elif self.strategy == "cos":
            self.step_size = lambda val: math.cos(math.pi / 2 * val / self.num_steps) * \
                                         (self.base_val - self.final_val) + self.final_val
        else:
            raise NotImplementedError("This decreasing policy is not supported")

    def step(self):
        if self.val <= self.final_val:
            return self.val
        if self.strategy == "linear":
            self.val -= self.step_size
        elif self.strategy == "exp":
            self.val *= self.step_size
        elif self.strategy == "cos":
            self.val = self.step_size(self.val)
        return self.val