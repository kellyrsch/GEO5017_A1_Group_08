"""Basic recursive gradient engine"""

from enum import Enum
from math import log
import sys
sys.setrecursionlimit(10000) # if we don't need this, we will very quickly reach Python's recursion limit

class Operation(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "**"

class Value:

    def __init__(self,
                 value: float | int,
                 _prev: tuple['Value', 'Value'] | None = None,
                 _prev_operation: Operation | None = None):
        self.val = value
        self.gradient = 0
        self._prev = _prev
        self._prev_op = _prev_operation

    def __key(self):
        return (self.val, self.gradient)

    def __repr__(self):
        return f"Value(val={self.val}, gradient={self.gradient})"
    
    def print_formula(self) -> str:
        if self._prev is None:
            return str(self.val)
        if self._prev[1] is None:
            ... # this would need to be implemented if we wanted to support functions such as tan, exp, log, cos, etc. (since these only have one value)
        return f"({self._prev[0].print_formula()}{self._prev_op.value}{self._prev[1].print_formula()})"
    
    def clear_gradient(self):
        self.gradient = 0
        if self._prev is None:
            return
        for p in self._prev:
            if p is not None:
                p.clear_gradient()

    def backpropagate(self):
        self.clear_gradient() # clear any existing gradients (since we will be incrementing gradients instead of overwriting them)
        self.gradient = 1 # we can do this because we are finding the derivative with respect to ourselves
        self._backpropagate()

    def recalculate(self):
        if self._prev_op is None:
            return self.val
        new_val = self.math(self._prev[0].recalculate(), self._prev[1].recalculate() if len(self._prev) > 1 else None, self._prev_op)
        self.val = new_val
        return new_val

    def _backpropagate(self):
        if self._prev is None:
            return
        non_none_prevs = [p for p in self._prev if p is not None]
        match self._prev_op:
            # these formulas are derived from basic calculus & the chain rule
            case Operation.ADD:
                self._prev[0].gradient += self.gradient # dx/dy[x + y] = 1
                self._prev[1].gradient += self.gradient # dy/dy[x + y] = 1
            case Operation.SUB:
                self._prev[0].gradient += self.gradient # dx/dy[x - y] = 1
                self._prev[1].gradient += -self.gradient # dy/dy[x - y] = -1
            case Operation.MUL:
                self._prev[0].gradient += self.gradient * self._prev[1].val # dx/dy[x * y] = y
                self._prev[1].gradient += self.gradient * self._prev[0].val # dy/dy[x * y] = x
            case Operation.DIV:
                self._prev[0].gradient += self.gradient * (self._prev[1].val ** -1) # dx/dy[x / y] = 1/y
                self._prev[1].gradient += self.gradient * (-self._prev[0].val * (self._prev[1].val ** -2)) # dy/dy[x / y] = -x/y^2
            case Operation.POW:
                self._prev[0].gradient += self._prev[1].val * (self._prev[0].val**(self._prev[1].val-1)) * self.gradient # dx/dy[x^y] = y*x^(y-1) (multiply by current gradient because of chain rule)
                self._prev[1].gradient += log(self._prev[1].val) * self._prev[1].val**self._prev[0].val * self.gradient # dy/dy[x^y] = x^y * log(x) (multiply by current gradient because of chain rule)
            case None:
                raise ValueError("Previous Operation can't be None if Previous Value isn't")
            case _:
                raise NotImplementedError(f"Unknown Operation {self._prev_op}")
        for p in non_none_prevs:
            p._backpropagate()

    def math(self, val_a, val_b, operation):
        match operation:
            case Operation.ADD:
                return val_a + val_b
            case Operation.SUB:
                return val_a - val_b
            case Operation.MUL:
                return val_a * val_b
            case Operation.DIV:
                return val_a / val_b
            case Operation.POW:
                return val_a ** val_b
            case None:
                raise ValueError("Cannot calculate with no operation")
            case _:
                raise NotImplementedError(f"Unknown Operation {self._prev_op}")
            
    def perform_operation(self, other, operation: Operation) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.math(self.val, other.val, operation), _prev_operation=operation, _prev=(self, other))

    def __add__(self, other) -> 'Value':
        return self.perform_operation(other, Operation.ADD)
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other) -> 'Value':
        return self.perform_operation(other, Operation.SUB)
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other) -> 'Value':
        return self.perform_operation(other, Operation.MUL)
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other) -> 'Value':
        val = other.val if isinstance(other, Value) else other
        assert isinstance(val, float) or isinstance(val, int)
        return self.perform_operation(other, Operation.POW)
    def __rpow__(self, other):
        return self.__pow__(other)
    
    def __truediv__(self, other) -> 'Value':
        return self.perform_operation(other, Operation.DIV)
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __eq__(self, other: 'Value') -> bool:
        assert isinstance(other, Value)
        return self.val == other.val # could we run into problems if two values have the same val but different gradients?
    
    def __lt__(self, other: 'Value') -> bool:
        assert isinstance(other, Value)
        return self.val < other.val
    
    def __hash__(self):
        return hash(self.__key())