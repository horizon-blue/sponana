from pydrake.all import DiagramBuilder, StateInterpolatorWithDiscreteDerivative


def make_spot_controller():
    """
    Create and return a Spot controller that output the state configuration
    (which can be feed into the hardware station to actually move the robot)
    """
    builder = DiagramBuilder()

    # Convert positions to desired state
    position_to_state = builder.AddNamedSystem(
        "position_to_state",
        StateInterpolatorWithDiscreteDerivative(
            # TODO: split the control of end effector (the gripper) from the
            # control of the base of Spot
            num_positions=10,
            time_step=0.05,
            suppress_initial_transient=True,
        ),
    )

    # Export I/O ports
    builder.ExportInput(position_to_state.get_input_port(), "position")
    builder.ExportOutput(position_to_state.get_output_port(), "desired_state")

    # Finalize the diagram
    spot_controller = builder.Build()
    spot_controller.set_name("spot_controller")
    return spot_controller


# alias, since we can think of the function as a "constructor"
SpotController = make_spot_controller
