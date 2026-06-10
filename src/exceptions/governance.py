class FinancialGovernanceViolation(Exception):
    """
    Custom core exception thrown when an ingress query, metadata profile,
    or retrieval chunk matrix violates strict data security guidelines.
    """
    def __init__(self, message: str = "Query dropped by financial data governance guardrails."):
        self.message = message
        super().__init__(self.message)