class JudgeResponse:
    def __init__(self, idx: int, points: int, explanation: str, detailed_cost: dict, history: list, additional_info: dict | None = None):
        self.idx = idx
        self.points = points
        self.explanation = explanation
        self.detailed_cost = detailed_cost
        self.history = history
        self.additional_info = additional_info or {}
