from pydrake.all import BasicVector, Context, LeafSystem


class PositionCombiner(LeafSystem):
    """A utility class that combine the position of the base of Spot (3 dims)
    and the position of the arm of Spot (7 dims) into a single position (10 dims).

    This can be useful when we want to separately control and solve for the
    positions of the base and the arm.
    """

    def __init__(self):
        super().__init__()
        # I/O
        # self.DeclareVectorInputPort("base_position", 3)
        # FIXME: allow it to be 10 for now becuase I havn't figured out how to get rid
        # of extra joints in the slider
        self.DeclareVectorInputPort("base_position", 10)
        self.DeclareVectorInputPort("arm_position", 7)
        self.DeclareVectorOutputPort("position", 10, self._combine_position)

    def get_base_position_input_port(self):
        return self.get_input_port(0)

    def get_arm_position_input_port(self):
        return self.get_input_port(1)

    def _combine_position(self, context: Context, output: BasicVector):
        base_position = self.get_base_position_input_port().Eval(context)[:3]
        arm_position = self.get_arm_position_input_port().Eval(context)
        output.SetFromVector([*base_position, *arm_position])
