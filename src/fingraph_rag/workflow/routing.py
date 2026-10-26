"""One shared repair budget across the whole request."""

MAX_REPAIRS = 2


def verification_route(state, max_repairs: int = MAX_REPAIRS):
    if state.get("terminal") == "answer":
        return "finish"
    if state.get("repairs", 0) >= max_repairs or not state.get("candidates"):
        return "finish"
    return "repair"
