def get_input_label_tuple(t, tuple_order):
    assert tuple_order in ("pl", "lp", "il", "li")

    if tuple_order in ("pl", "il"):
        input, label = t
    elif tuple_order in ("lp", "li"):
        label, input = t
    else:
        raise ValueError()

    return input, label