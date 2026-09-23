"""Column labels shared by CSV export and visualization."""


def component_label(key, index, barrier_labels):
    if key == "ee_yaw":
        return "ee_yaw [rad]"
    if key == "barrier" and index < len(barrier_labels):
        return barrier_labels[index]
    return f"{key}_{index}"
