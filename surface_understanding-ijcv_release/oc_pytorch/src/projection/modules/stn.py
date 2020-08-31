from torch.nn.modules.module import Module
from projection.functions.stn import STNFunction
from projection.functions.stn import VTNFunction

class STN(Module):
    def __init__(self):
        super(STN, self).__init__()
        self.f = STNFunction()
    def forward(self, input1, input2):
        return self.f(input1, input2)

class VTN(Module):
    def __init__(self):
        super(VTN, self).__init__()
        self.f = VTNFunction()
    def forward(self, input1, input2):
        return self.f(input1, input2)
