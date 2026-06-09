class FinancialGovernanceViolation(Exception):
    """Custom application exception raised when corporate compliance perimeters are violated."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)